import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------------
# 1. Load and preprocess data
# -----------------------------------
with open("data.json", "r") as f:
    raw_data = json.load(f)

# Tokenization
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build vocab and template set
PAD, UNK = "<PAD>", "<UNK>"
token_counter = Counter()
template_set = set()
for item in raw_data:
    tokens = tokenize(item["question"])
    token_counter.update(tokens)
    tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
    tpl = re.sub(r"\s+", " ", tpl.strip())
    template_set.add(tpl)

vocab = {PAD: 0, UNK: 1}
for tok in token_counter:
    vocab[tok] = len(vocab)

template2id = {tpl: i for i, tpl in enumerate(sorted(template_set))}
id2template = {i: tpl for tpl, i in template2id.items()}

# -----------------------------------
# 2. BIO tag generation for multi-word cities
# -----------------------------------
tag2id = {"O": 0, "B-CITY": 1, "I-CITY": 2}
id2tag = {v: k for k, v in tag2id.items()}

def generate_bio_tags_for_question(question: str, sql: str):
    tokens = tokenize(question)
    tags = ["O"] * len(tokens)
    # extract real city names
    city_names = re.findall(r'CITY_NAME\s*=\s*"([^"]+)"', sql)
    # placeholders in question
    placeholders = re.findall(r'(city_name\d+)', question.lower())
    for ph, city in zip(placeholders, city_names):
        city_tokens = city.lower().split()
        try:
            idx = tokens.index(ph)
        except ValueError:
            continue
        tags[idx] = "B-CITY"
        for j in range(1, len(city_tokens)):
            if idx + j < len(tags):
                tags[idx + j] = "I-CITY"
    return tokens, [tag2id[t] for t in tags]

# -----------------------------------
# 3. Joint Dataset and DataLoader
# -----------------------------------
class JointDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens, tag_ids = generate_bio_tags_for_question(item["question"], item["sql"])
            input_ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            tpl = re.sub(r"\s+", " ", tpl.strip())
            tmpl_id = template2id[tpl]
            self.samples.append((input_ids, tmpl_id, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# pad batch

def pad_batch(batch):
    input_ids, tmpl_ids, tag_ids = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    pad_seq = lambda seq: seq + [vocab[PAD]] * (max_len - len(seq))
    pad_tag = lambda seq: seq + [tag2id["O"]] * (max_len - len(seq))
    inputs = torch.tensor([pad_seq(x) for x in input_ids])
    templates = torch.tensor(tmpl_ids)
    tags = torch.tensor([pad_tag(t) for t in tag_ids])
    return inputs, templates, tags

# -----------------------------------
# 4. Transformer model definition
# -----------------------------------
class TransformerClassifierTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_templates, num_tags, pad_idx=0, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.template_classifier = nn.Linear(embedding_dim, num_templates)
        self.token_tagger = nn.Linear(embedding_dim, num_tags)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        encoded = self.transformer_encoder(x)
        cls_out = encoded[:, 0, :]
        template_logits = self.template_classifier(cls_out)
        tag_logits = self.token_tagger(encoded)
        return template_logits, tag_logits

# -----------------------------------
# 5. Training pipeline
# -----------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train/val split
    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    train_ds = JointDataset(train_data, vocab, template2id)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pad_batch)

    model = TransformerClassifierTagger(
        vocab_size=len(vocab),
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        num_templates=len(template2id),
        num_tags=len(tag2id)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_tmpl = nn.CrossEntropyLoss()
    loss_fn_tag  = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss, correct = 0.0, 0
        for x_batch, y_tmpl, y_tags in train_loader:
            x_batch, y_tmpl, y_tags = x_batch.to(device), y_tmpl.to(device), y_tags.to(device)
            pred_tmpl, pred_tags = model(x_batch)
            loss1 = loss_fn_tmpl(pred_tmpl, y_tmpl)
            B, T, C = pred_tags.size()
            loss2 = loss_fn_tag(pred_tags.view(B*T, C), y_tags.view(B*T))
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B
            correct += (pred_tmpl.argmax(1) == y_tmpl).sum().item()
        print(f"[Epoch {epoch+1}] Loss={total_loss/len(train_ds):.4f} Template Acc={correct/len(train_ds):.4f}")
