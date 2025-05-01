# train_transformer.py

import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define Transformer model here since kernel was reset
class TransformerClassifierTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_templates, num_tags, pad_idx=0, max_len=128):
        super(TransformerClassifierTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.template_classifier = nn.Linear(embedding_dim, num_templates)
        self.token_tagger = nn.Linear(embedding_dim, num_tags)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        encoded = self.transformer_encoder(x)
        cls_output = encoded[:, 0, :]
        template_logits = self.template_classifier(cls_output)
        tag_logits = self.token_tagger(encoded)
        return template_logits, tag_logits

# === Step 1: Load and preprocess data ===
with open("data.json", "r") as f:
    raw_data = json.load(f)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

PAD, UNK = "<PAD>", "<UNK>"
token_counter = Counter()
template_set = set()

for item in raw_data:
    tokens = tokenize(item["question"])
    token_counter.update(tokens)
    sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
    sql_template = re.sub(r"\s+", " ", sql_template.strip())
    template_set.add(sql_template)

vocab = {PAD: 0, UNK: 1}
for word in token_counter:
    vocab[word] = len(vocab)

template2id = {tpl: i for i, tpl in enumerate(sorted(template_set))}
tag2id = {"O": 0}

# === Step 2: Dataset and Dataloader ===
class TransformerDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            input_ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            sql_template = re.sub(r'\s+', ' ', sql_template.strip())
            tid = template2id[sql_template]
            tag_ids = [0] * len(tokens)  # Placeholder
            self.samples.append((input_ids, tid, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def pad_batch(batch):
    input_ids, template_ids, tag_ids = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    pad_input = lambda seq: seq + [vocab[PAD]] * (max_len - len(seq))
    pad_tags = lambda seq: seq + [tag2id["O"]] * (max_len - len(seq))
    inputs = torch.tensor([pad_input(seq) for seq in input_ids])
    tags = torch.tensor([pad_tags(seq) for seq in tag_ids])
    templates = torch.tensor(template_ids)
    return inputs, templates, tags

train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)
train_loader = DataLoader(TransformerDataset(train_data, vocab, template2id), batch_size=16, shuffle=True, collate_fn=pad_batch)
val_loader = DataLoader(TransformerDataset(val_data, vocab, template2id), batch_size=16, shuffle=False, collate_fn=pad_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifierTagger(
    vocab_size=len(vocab),
    embedding_dim=128,
    num_heads=4,
    num_layers=2,
    num_templates=len(template2id),
    num_tags=len(tag2id)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_template = nn.CrossEntropyLoss()
loss_tags = nn.CrossEntropyLoss()

# === Step 3: Training loop ===
for epoch in range(5):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y_template, y_tags in train_loader:
        x, y_template, y_tags = x.to(device), y_template.to(device), y_tags.to(device)
        pred_template, pred_tags = model(x)
        loss1 = loss_template(pred_template, y_template)
        loss2 = loss_tags(pred_tags.view(-1, pred_tags.shape[-1]), y_tags.view(-1))
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (pred_template.argmax(1) == y_template).sum().item()
    acc = total_correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Template Acc: {acc:.4f}")
