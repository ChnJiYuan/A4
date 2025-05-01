# Re-running train_ff.py after kernel reset

import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define FF models again since the kernel was reset
class FFTemplateClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_templates, pad_idx=0):
        super(FFTemplateClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_templates)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        mean_emb = emb.mean(dim=1)
        return self.ff(mean_emb)

class FFSequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, pad_idx=0):
        super(FFSequenceTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tags)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        return self.ff(emb)

# Data loading
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

class FFDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            input_ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            sql_template = re.sub(r'\s+', ' ', sql_template.strip())
            tid = template2id[sql_template]
            tag_ids = [0] * len(tokens)  # placeholder tags
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
train_loader = DataLoader(FFDataset(train_data, vocab, template2id), batch_size=16, shuffle=True, collate_fn=pad_batch)
val_loader = DataLoader(FFDataset(val_data, vocab, template2id), batch_size=16, shuffle=False, collate_fn=pad_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
template_model = FFTemplateClassifier(len(vocab), 100, 128, len(template2id)).to(device)
tag_model = FFSequenceTagger(len(vocab), 100, 128, len(tag2id)).to(device)

opt_template = optim.Adam(template_model.parameters(), lr=0.001)
opt_tags = optim.Adam(tag_model.parameters(), lr=0.001)

loss_template = nn.CrossEntropyLoss()
loss_tags = nn.CrossEntropyLoss()

for epoch in range(5):
    template_model.train()
    tag_model.train()
    total_loss, total_correct = 0, 0
    for x, y_template, y_tags in train_loader:
        x, y_template, y_tags = x.to(device), y_template.to(device), y_tags.to(device)

        pred_template = template_model(x)
        loss1 = loss_template(pred_template, y_template)
        opt_template.zero_grad()
        loss1.backward()
        opt_template.step()

        pred_tags = tag_model(x)
        loss2 = loss_tags(pred_tags.view(-1, pred_tags.shape[-1]), y_tags.view(-1))
        opt_tags.zero_grad()
        loss2.backward()
        opt_tags.step()

        total_loss += loss1.item() + loss2.item()
        total_correct += (pred_template.argmax(1) == y_template).sum().item()

    acc = total_correct / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Template Acc: {acc:.4f}")
