import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# -----------------------------------
# 1. Load and preprocess data
# -----------------------------------
with open("data.json", "r") as f:
    data = json.load(f)

# Tokenization function
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build vocabulary
PAD, UNK = "<PAD>", "<UNK>"
token_counter = Counter()
for item in data:
    token_counter.update(tokenize(item["question"]))
vocab = {PAD: 0, UNK: 1}
for token in token_counter:
    vocab[token] = len(vocab)

# Build SQL template mapping
template_counter = Counter()
for item in data:
    tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
    tpl = re.sub(r"\s+", " ", tpl.strip())
    template_counter[tpl] += 1
template2id = {tpl: i for i, tpl in enumerate(template_counter)}
id2template = {i: tpl for tpl, i in template2id.items()}

# -----------------------------------
# 2. BIO tags for multi-word cities & multi-occurrence placeholders
# -----------------------------------
tag2id = {"O": 0, "B-CITY": 1, "I-CITY": 2}
id2tag = {v: k for k, v in tag2id.items()}

def generate_bio_tags_for_question(question: str, sql: str):
    tokens = tokenize(question)
    tags = ["O"] * len(tokens)

    # Extract city names from SQL and placeholders in question
    cities = re.findall(r'CITY_NAME\s*=\s*"([^"]+)"', sql)
    placeholders = re.findall(r'(city_name\d+)', question.lower())

    # 1) Annotate all placeholder occurrences
    for ph, city in zip(placeholders, cities):
        city_tokens = city.lower().split()
        L = len(city_tokens)
        for i, tok in enumerate(tokens):
            if tok == ph and tags[i] == "O":
                tags[i] = "B-CITY"
                for j in range(1, L):
                    if i + j < len(tags) and tags[i + j] == "O":
                        tags[i + j] = "I-CITY"

    # 2) Sliding-window match for literal multi-word city names
    for city in cities:
        city_tokens = city.lower().split()
        L = len(city_tokens)
        for i in range(len(tokens) - L + 1):
            if tokens[i:i+L] == city_tokens and all(tags[i + j] == "O" for j in range(L)):
                tags[i] = "B-CITY"
                for j in range(1, L):
                    tags[i + j] = "I-CITY"

    return tokens, [tag2id[t] for t in tags]

# -----------------------------------
# 3. Dataset definitions
# -----------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            ids = [vocab.get(t, vocab[UNK]) for t in tokens]
            tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            tpl = re.sub(r"\s+", " ", tpl.strip())
            tid = template2id[tpl]
            self.samples.append((ids, tid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TaggingDataset(Dataset):
    def __init__(self, data, vocab):
        self.samples = []
        for item in data:
            tokens, tag_ids = generate_bio_tags_for_question(item["question"], item["sql"])
            ids = [vocab.get(t, vocab[UNK]) for t in tokens]
            self.samples.append((ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Collate functions

def collate_class(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    inputs_padded = [x + [vocab[PAD]]*(max_len-len(x)) for x in inputs]
    return torch.tensor(inputs_padded), torch.tensor(labels)

def collate_tag(batch):
    inputs, tags = zip(*batch)
    max_len = max(len(x) for x in inputs)
    inputs_padded = [x + [vocab[PAD]]*(max_len-len(x)) for x in inputs]
    tags_padded   = [t + [tag2id["O"]]*(max_len-len(t)) for t in tags]
    return torch.tensor(inputs_padded), torch.tensor(tags_padded)

# -----------------------------------
# 4. Model definitions
# -----------------------------------
class FFTemplateClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_templates, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_templates)
        )

    def forward(self, x):
        emb = self.embedding(x)          # (B, T, E)
        pooled = emb.mean(dim=1)         # (B, E)
        return self.ff(pooled)           # (B, num_templates)

class FFSequenceTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_tags, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_tags)
        )

    def forward(self, x):
        emb = self.embedding(x)          # (B, T, E)
        return self.ff(emb)              # (B, T, num_tags)

# -----------------------------------
# 5. Training loops
# -----------------------------------
if __name__ == "__main__":
    # Classification
    clf_ds = ClassificationDataset(data, vocab, template2id)
    clf_dl = DataLoader(clf_ds, batch_size=16, shuffle=True, collate_fn=collate_class)
    clf_model = FFTemplateClassifier(len(vocab), 100, 128, len(template2id)).to("cpu")
    clf_opt   = optim.Adam(clf_model.parameters(), lr=1e-3)
    clf_loss  = nn.CrossEntropyLoss()
    for epoch in range(5):
        clf_model.train()
        tot_loss, corr, tot = 0, 0, 0
        for x, y in clf_dl:
            logits = clf_model(x)
            loss = clf_loss(logits, y)
            clf_opt.zero_grad(); loss.backward(); clf_opt.step()
            tot_loss += loss.item()*y.size(0)
            corr += (logits.argmax(1)==y).sum().item()
            tot += y.size(0)
        print(f"[Clf] Epoch {epoch+1} Loss={tot_loss/tot:.4f} Acc={corr/tot:.4f}")

    # Tagging
    tag_ds = TaggingDataset(data, vocab)
    tag_dl = DataLoader(tag_ds, batch_size=16, shuffle=True, collate_fn=collate_tag)
    tag_model = FFSequenceTagger(len(vocab), 100, 128, len(tag2id)).to("cpu")
    tag_opt   = optim.Adam(tag_model.parameters(), lr=1e-3)
    tag_loss  = nn.CrossEntropyLoss()
    for epoch in range(5):
        tag_model.train()
        tot_loss, tokens = 0, 0
        for x, y in tag_dl:
            logits = tag_model(x)
            B, T, C = logits.shape
            loss = tag_loss(logits.view(B*T, C), y.view(B*T))
            tag_opt.zero_grad(); loss.backward(); tag_opt.step()
            tot_loss += loss.item()*B*T
            tokens += B*T
        print(f"[Tag] Epoch {epoch+1} Loss={tot_loss/tokens:.4f}")
