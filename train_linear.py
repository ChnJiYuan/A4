import json
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.polys.rootisolation import A4
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import re
import random

# Load data
with open("data.json", "r") as f:
    data = json.load(f)

# Tokenize and normalize
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build vocab from questions
token_counter = Counter()
for item in data:
    token_counter.update(tokenize(item["question"]))

PAD, UNK = "<PAD>", "<UNK>"
vocab = {PAD: 0, UNK: 1}
for token, _ in token_counter.items():
    vocab[token] = len(vocab)

# Build SQL template ID map
template_counter = Counter()
for item in data:
    sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])  # mask values
    sql_template = re.sub(r'\s+', ' ', sql_template.strip())
    template_counter[sql_template] += 1

template2id = {tpl: i for i, tpl in enumerate(template_counter)}
id2template = {i: tpl for tpl, i in template2id.items()}

# Generate classification dataset
class ClassificationDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            sql_template = re.sub(r'\s+', ' ', sql_template.strip())
            tid = template2id[sql_template]
            self.samples.append((ids, tid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Collate function for padding
def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded = [x + [vocab[PAD]] * (max_len - len(x)) for x in inputs]
    return torch.tensor(padded), torch.tensor(labels)

# Linear Model for Template Classification
class LinearTemplateClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_templates):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, num_templates)

    def forward(self, x):
        emb = self.embedding(x)  # (B, T, E)
        pooled = emb.mean(dim=1)  # (B, E)
        return self.linear(pooled)  # (B, num_templates)

# Prepare training
dataset = ClassificationDataset(data, vocab, template2id)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

model = LinearTemplateClassifier(vocab_size=len(vocab), emb_dim=100, num_templates=len(template2id))
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for x, y in train_loader:
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    acc = total_correct / total
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Acc = {acc:.4f}")

