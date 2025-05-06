import json
import re
import argparse
import subprocess
import platform
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Enable CuDNN benchmark for potential speedup on fixed-size inputs
torch.backends.cudnn.benchmark = True

# -------------------------------------------
# Device configuration
# -------------------------------------------
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------
# Data utilities
# -------------------------------------------

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text):
    return text.lower().split()


class ATISDataset(Dataset):
    def __init__(self, split_json, template_map, tag_map, vocab, max_len=50):
        # Load examples
        with open(split_json, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        print(f"Preparing dataset from {split_json}, {len(self.examples)} examples")
        self.templates = template_map
        self.tags = tag_map
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        # Iterate with progress bar
        for ex in tqdm(self.examples, desc=f"Building {split_json}"):
            text = normalize_whitespace(ex['text'])
            tpl = ex['sql']
            template_id = self._match_template_id(tpl)
            tokens = tokenize(text)
            tag_seq = self._generate_tags(tokens, ex)
            # Pad or truncate
            if len(tokens) < max_len:
                tokens += ['<pad>'] * (max_len - len(tokens))
                tag_seq += [self.tags['O']] * (max_len - len(tag_seq))
            else:
                tokens = tokens[:max_len]
                tag_seq = tag_seq[:max_len]
            self.data.append((tokens, tag_seq, template_id))
        print(f"Finished building dataset from {split_json}")

    def _match_template_id(self, sql_with_vals):
        # Find template id by regex match
        for tpl, tid in self.templates.items():
            pattern = re.escape(tpl)
            pattern = re.sub(r"\\\"(.*?)\\\"", r'".+?"', pattern)
            if re.fullmatch(pattern, sql_with_vals):
                return tid
        return -1

    def _generate_tags(self, tokens, ex):
        # Tag tokens according to variables
        tags = ['O'] * len(tokens)
        for var, val in ex.get('variables', {}).items():
            val_tokens = tokenize(val)
            for i in range(len(tokens) - len(val_tokens) + 1):
                if tokens[i:i + len(val_tokens)] == val_tokens:
                    for j in range(len(val_tokens)):
                        tags[i + j] = var
        return [self.tags.get(t, self.tags['O']) for t in tags]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tag_seq, template_id = self.data[idx]
        input_ids = [self.vocab.get(tok, self.vocab['<unk>']) for tok in tokens]
        return torch.tensor(input_ids), torch.tensor(tag_seq), torch.tensor(template_id)

# -------------------------------------------
# Model definition
# -------------------------------------------

class LSTMTaggerTemplate(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_templates, n_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.tag_fc = nn.Linear(hidden_dim, n_tags)
        self.template_fc = nn.Linear(hidden_dim, n_templates)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, _) = self.lstm(emb)
        tag_logits = self.tag_fc(out)
        final_h = h_n[-1]
        temp_logits = self.template_fc(final_h)
        return tag_logits, temp_logits

# -------------------------------------------
# Helpers to build maps and vocab
# -------------------------------------------

def build_maps(train_json, atis_json):
    with open(atis_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    templates = sorted({normalize_whitespace(sorted(entry['sql'], key=lambda x: (len(x), x))[0]) for entry in data})
    template_map = {tpl: i for i, tpl in enumerate(templates)}
    tag_set = {var for ex in json.load(open(train_json, 'r', encoding='utf-8')) for var in ex.get('variables', {})}
    tag_map = {'O': 0, **{var: i+1 for i, var in enumerate(sorted(tag_set))}}
    return template_map, tag_map


def build_vocab(train_json, test_json):
    counter = Counter()
    for path in (train_json, test_json):
        for ex in json.load(open(path, 'r', encoding='utf-8')):
            counter.update(tokenize(normalize_whitespace(ex['text'])))
    vocab = {'<pad>': 0, '<unk>': 1}
    for tok, _ in counter.most_common():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

# -------------------------------------------
# Training and evaluation
# -------------------------------------------

def train_model(args):
    # Build templates, tags, vocab
    template_map, tag_map = build_maps(args.train_split, args.atis)
    vocab = build_vocab(args.train_split, args.test_split)

    # Prepare datasets
    train_ds = ATISDataset(args.train_split, template_map, tag_map, vocab)
    test_ds = ATISDataset(args.test_split, template_map, tag_map, vocab)

    # DataLoader config
    num_workers = 0 if platform.system() == 'Windows' else (4 if use_cuda else 0)
    loader_args = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': num_workers,
                   'pin_memory': use_cuda}
    train_loader = DataLoader(train_ds, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=1,
                             num_workers=num_workers,
                             pin_memory=use_cuda)

    # Model, loss, optimizer
    model = LSTMTaggerTemplate(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_templates=len(template_map),
        n_tags=len(tag_map)
    ).to(device)
    tag_loss_fn = nn.CrossEntropyLoss(ignore_index=tag_map['O'])
    temp_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        for input_ids, tag_seqs, temp_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"         , unit="batch"):
            input_ids, tag_seqs, temp_ids = input_ids.to(device), tag_seqs.to(device), temp_ids.to(device)
            optimizer.zero_grad()
            tag_logits, temp_logits = model(input_ids)
            B, L, _ = tag_logits.size()
            loss = tag_loss_fn(tag_logits.view(B*L, -1), tag_seqs.view(-1)) + temp_loss_fn(temp_logits, temp_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} done, Avg Loss: {avg_loss:.4f}\n")

    # Inference
    model.eval()
    preds = []
    templates_inv = {v: k for k, v in template_map.items()}
    print("Starting inference on test set...")
    for input_ids, tag_seqs, temp_ids in tqdm(test_loader, desc="Inference", unit="ex"):
        input_ids = input_ids.to(device)
        tag_logits, temp_logits = model(input_ids)
        tpl = templates_inv[temp_logits.argmax(dim=-1).item()]
        tag_ids = tag_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        tokens = [list(vocab.keys())[list(vocab.values()).index(id_)] for id_ in input_ids.squeeze(0).cpu().tolist()]
        var_values = defaultdict(list)
        for tok, tid in zip(tokens, tag_ids):
            name = next((n for n, idx in tag_map.items() if idx == tid), 'O')
            if name != 'O' and tok != '<pad>':
                var_values[name].append(tok)
        sql = tpl
        for var, toks in var_values.items():
            sql = sql.replace(f'"{var}"', f'"{' '.join(toks).upper()}"')
        preds.append(normalize_whitespace(sql))

    # Save predictions
    with open(args.predictions, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(p + '\n')

    # Evaluate
    accuracy_cmd = ["python", args.accuracy_path,
                    "--predictions", args.predictions,
                    "--atis", args.atis,
                    "--split", args.test_split]
    if args.shortest:
        accuracy_cmd.append("--shortest")
    subprocess.run(accuracy_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM classification for ATIS')
    parser.add_argument('--atis', required=True, help='Path to atis.json')
    parser.add_argument('--train_split', required=True, help='Path to train split JSON')
    parser.add_argument('--test_split', required=True, help='Path to test split JSON')
    parser.add_argument('--predictions', default='predictions.txt', help='File to write SQL predictions')
    parser.add_argument('--accuracy_path', default='Accuracy.py', help='Path to Accuracy.py script')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--shortest', action='store_true', help='Match only shortest SQL')
    args = parser.parse_args()
    train_model(args)
