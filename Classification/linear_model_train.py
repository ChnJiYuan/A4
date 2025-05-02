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
with open("data.json", "r", encoding="utf-8") as f:
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

# Build SQL template to ID map
template_counter = Counter()
for item in data:
    sql_template = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
    sql_template = re.sub(r"\s+", " ", sql_template.strip())
    template_counter[sql_template] += 1
template2id = {tpl: i for i, tpl in enumerate(template_counter)}
id2template = {i: tpl for tpl, i in template2id.items()}

# -----------------------------------
# 2. BIO tag generation for cities (multi-occurrence support)
# -----------------------------------
tag2id = {"O": 0, "B-CITY": 1, "I-CITY": 2}
id2tag = {v: k for k, v in tag2id.items()}

def generate_bio_tags_for_question(question: str, sql: str):
    tokens = tokenize(question)
    tags = ["O"] * len(tokens)

    # 2.1 Extract all city names from SQL
    city_names = re.findall(r'CITY_NAME\s*=\s*"([^"]+)"', sql)
    # 2.2 Find placeholders in question
    placeholders = re.findall(r'(city_name\d+)', question.lower())

    # 2.3 Annotate all placeholder occurrences
    for ph, city in zip(placeholders, city_names):
        city_tokens = city.lower().split()
        L = len(city_tokens)
        for i in range(len(tokens) - L + 1):
            if tokens[i] == ph and tags[i] == "O":
                tags[i] = "B-CITY"
                for j in range(1, L):
                    if i + j < len(tags) and tags[i+j] == "O":
                        tags[i+j] = "I-CITY"

    # 2.4 Sliding-window match for literal multi-word city names
    for city in city_names:
        city_tokens = city.lower().split()
        L = len(city_tokens)
        for i in range(len(tokens) - L + 1):
            if tokens[i:i+L] == city_tokens and all(tags[i+j] == "O" for j in range(L)):
                tags[i] = "B-CITY"
                for j in range(1, L):
                    tags[i+j] = "I-CITY"

    return tokens, [tag2id[t] for t in tags]

# -----------------------------------
# 3. Dataset definitions
# -----------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens = tokenize(item["question"])
            ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            sql_tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            sql_tpl = re.sub(r"\s+", " ", sql_tpl.strip())
            tid = template2id[sql_tpl]
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
            ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            self.samples.append((ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Collate functions
def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded = [x + [vocab[PAD]] * (max_len - len(x)) for x in inputs]
    return torch.tensor(padded), torch.tensor(labels)

def collate_tagging(batch):
    inputs, tags = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded_inputs = [x + [vocab[PAD]] * (max_len - len(x)) for x in inputs]
    padded_tags   = [t + [tag2id["O"]] * (max_len - len(t)) for t in tags]
    return torch.tensor(padded_inputs), torch.tensor(padded_tags)

# -----------------------------------
# 4. Model definitions
# -----------------------------------
class LinearTemplateClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_templates):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, num_templates)

    def forward(self, x):
        emb = self.embedding(x)
        pooled = emb.mean(dim=1)
        return self.linear(pooled)

class LinearSequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, num_tags)

    def forward(self, x):
        emb = self.embedding(x)
        logits = self.linear(emb)
        return logits

# -----------------------------------
# 5. Training pipeline
# -----------------------------------
if __name__ == "__main__":
    # Classification training
    clf_dataset = ClassificationDataset(data, vocab, template2id)
    clf_loader  = DataLoader(clf_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    clf_model = LinearTemplateClassifier(len(vocab), embedding_dim=100, num_templates=len(template2id))
    clf_optimizer = optim.Adam(clf_model.parameters(), lr=0.001)
    clf_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        clf_model.train()
        total_loss, total_corr, total = 0.0, 0, 0
        for x_batch, y_batch in clf_loader:
            logits = clf_model(x_batch)
            loss = clf_loss_fn(logits, y_batch)
            clf_optimizer.zero_grad()
            loss.backward()
            clf_optimizer.step()
            total_loss += loss.item() * y_batch.size(0)
            total_corr += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
        print(f"[Classification] Epoch {epoch+1}: Loss={total_loss/total:.4f}, Acc={total_corr/total:.4f}")

    # Tagging training
    tag_dataset = TaggingDataset(data, vocab)
    tag_loader  = DataLoader(tag_dataset, batch_size=16, shuffle=True, collate_fn=collate_tagging)

    tag_model = LinearSequenceTagger(len(vocab), embedding_dim=100, num_tags=len(tag2id))
    tag_optimizer = optim.Adam(tag_model.parameters(), lr=0.001)
    tag_loss_fn   = nn.CrossEntropyLoss()

    for epoch in range(5):
        tag_model.train()
        total_loss = 0.0
        total_tokens = 0
        for x_batch, y_batch in tag_loader:
            logits = tag_model(x_batch)
            B, T, C = logits.shape
            loss = tag_loss_fn(logits.view(B*T, C), y_batch.view(B*T))
            tag_optimizer.zero_grad()
            loss.backward()
            tag_optimizer.step()
            total_loss += loss.item() * B * T
            total_tokens += B * T
        print(f"[Tagging] Epoch {epoch+1}: Loss={total_loss/total_tokens:.4f}")
