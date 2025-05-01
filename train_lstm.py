# train_lstm.py

import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

# === Step 1: Load and preprocess data ===
with open("data.json", "r") as f:
    raw_data = json.load(f)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

PAD, UNK = "<PAD>", "<UNK>"

# Build vocab
token_counter = Counter()
tag_counter = Counter()
template_set = set()

for item in raw_data:
    tokens = tokenize(item["question"])
    token_counter.update(tokens)

    sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
    sql_template = re.sub(r"\s+", " ", sql_template.strip())
    template_set.add(sql_template)

    # Mock variable tags for now (in real use, you'd need pre-tagged labels)
    tag_counter.update(["O"] * len(tokens))  # Replace this with true tags if available

# Vocab and mappings
vocab = {PAD: 0, UNK: 1}
for word in token_counter:
    vocab[word] = len(vocab)

template2id = {tpl: i for i, tpl in enumerate(sorted(template_set))}
id2template = {i: tpl for tpl, i in template2id.items()}

tag2id = {"O": 0}
id2tag = {0: "O"}

# === Step 2: Dataset and Dataloader ===
class LSTMDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            input_ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]

            sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            sql_template = re.sub(r'\s+', ' ', sql_template.strip())
            tid = template2id[sql_template]

            tag_ids = [0] * len(tokens)  # placeholder tags ("O")

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

# === Step 3: Model ===
from lstm_model import LSTMClassifierTagger

# === Step 4: Training ===
train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)
train_dataset = LSTMDataset(train_data, vocab, template2id)
val_dataset = LSTMDataset(val_data, vocab, template2id)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=pad_batch)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifierTagger(vocab_size=len(vocab),
                             embedding_dim=100,
                             hidden_dim=128,
                             num_templates=len(template2id),
                             num_tags=len(tag2id)).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_template = nn.CrossEntropyLoss()
loss_tags = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss, total_acc = 0, 0
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
        total_acc += (pred_template.argmax(1) == y_template).sum().item()

    acc = total_acc / len(train_dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Template Acc: {acc:.4f}")

