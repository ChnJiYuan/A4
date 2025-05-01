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
# 2. BIO tag generation for cities
# -----------------------------------
tag2id = {"O": 0, "B-CITY": 1, "I-CITY": 2}
id2tag = {v: k for k, v in tag2id.items()}

def generate_bio_tags_for_question(question: str, sql: str):
    tokens = tokenize(question)
    tags = ["O"] * len(tokens)

    # Extract real city names from SQL
    city_names = re.findall(r'CITY_NAME\s*=\s*"([^"]+)"', sql)
    # Extract placeholders from question
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
# 3. Dataset and DataLoader
# -----------------------------------
class JointDataset(Dataset):
    def __init__(self, data, vocab, template2id):
        self.samples = []
        for item in data:
            tokens, tag_ids = generate_bio_tags_for_question(item["question"], item["sql"])
            input_ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
            tpl = re.sub(r'"[^"]+"', '"VAR"', item["sql"])
            tpl = re.sub(r"\s+", " ", tpl.strip())
            tid = template2id[tpl]
            self.samples.append((input_ids, tid, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    inputs, template_ids, tag_ids = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded_inputs = [x + [vocab[PAD]] * (max_len - len(x)) for x in inputs]
    padded_tags   = [t + [tag2id["O"]] * (max_len - len(t)) for t in tag_ids]
    return torch.tensor(padded_inputs), torch.tensor(template_ids), torch.tensor(padded_tags)

# -----------------------------------
# 4. Model definition
# -----------------------------------
class LSTMClassifierTagger(nn.Module):
    """
    A single LSTM model that jointly:
    - Classifies the input question into a SQL template
    - Predicts a tag for each token
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_templates, num_tags, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.template_classifier = nn.Linear(hidden_dim, num_templates)
        self.token_tagger = nn.Linear(hidden_dim, num_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        outputs, (h_n, _) = self.lstm(embedded)
        final_hidden = h_n.squeeze(0)
        template_logits = self.template_classifier(final_hidden)
        tag_logits = self.token_tagger(outputs)
        return template_logits, tag_logits

# -----------------------------------
# 5. Training pipeline
# -----------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = JointDataset(data, vocab, template2id)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = LSTMClassifierTagger(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_templates=len(template2id),
        num_tags=len(tag2id)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_template = nn.CrossEntropyLoss()
    loss_tags = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        total_correct = 0
        for x_batch, tmpl_batch, tag_batch in loader:
            x_batch = x_batch.to(device)
            tmpl_batch = tmpl_batch.to(device)
            tag_batch = tag_batch.to(device)

            pred_tmpl, pred_tags = model(x_batch)
            loss1 = loss_template(pred_tmpl, tmpl_batch)
            B, T, C = pred_tags.shape
            loss2 = loss_tags(pred_tags.view(B * T, C), tag_batch.view(B * T))
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_correct += (pred_tmpl.argmax(1) == tmpl_batch).sum().item()

        acc = total_correct / len(dataset)
        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}] Loss={avg_loss:.4f} | Template Acc={acc:.4f}")
