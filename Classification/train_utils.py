import re
import torch
from collections import Counter

def tokenize(text: str) -> list[str]:
    """
    Simple word-level tokenizer: splits on word boundaries and lowers case.
    """
    return re.findall(r"\b\w+\b", text.lower())

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(data: list[dict], min_freq: int = 1) -> dict:
    """
    Build a token->index vocab from a list of examples.
    Keeps PAD and UNK tokens, then includes any token with frequency >= min_freq.
    data: each item must have a 'question' field.
    """
    counter = Counter()
    for item in data:
        counter.update(tokenize(item['question']))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def build_template2id(data: list[dict]) -> dict:
    """
    Build a mapping from normalized SQL template to unique ID.
    Normalization: replace all literal values ("...") with "VAR" and collapse whitespace.
    """
    templates = set()
    for item in data:
        tpl = re.sub(r'"[^"]+"', '"VAR"', item['sql'])
        tpl = re.sub(r"\s+", ' ', tpl.strip())
        templates.add(tpl)
    return {tpl: i for i, tpl in enumerate(sorted(templates))}


def generate_bio_tags_for_question(question: str, sql: str) -> tuple[list[str], list[int]]:
    """
    Generate BIO tags for multi-word city placeholders and literal city names.
    Tags: O, B-CITY, I-CITY. Returns token list and tag id list.
    Requires tag2id mapping in calling scope.
    """
    tokens = tokenize(question)
    tags = ['O'] * len(tokens)

    # extract city names from SQL
    city_names = re.findall(r'CITY_NAME\s*=\s*"([^"]+)"', sql)
    # extract placeholders in question
    placeholders = re.findall(r'(city_name\d+)', question.lower())

    # mark placeholder matches
    for ph, city in zip(placeholders, city_names):
        city_toks = city.lower().split()
        L = len(city_toks)
        for i in range(len(tokens) - L + 1):
            if tokens[i] == ph and tags[i] == 'O':
                tags[i] = 'B-CITY'
                for j in range(1, L):
                    if tags[i+j] == 'O':
                        tags[i+j] = 'I-CITY'

    # mark literal city name spans
    for city in city_names:
        city_toks = city.lower().split()
        L = len(city_toks)
        for i in range(len(tokens) - L + 1):
            if tokens[i:i+L] == city_toks and all(tags[i+j] == 'O' for j in range(L)):
                tags[i] = 'B-CITY'
                for j in range(1, L):
                    tags[i+j] = 'I-CITY'

    return tokens, tags


def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string by collapsing whitespace.
    """
    return ' '.join(sql.strip().split())


def fill_template(sql_template: str, var_mapping: dict) -> str:
    """
    Replace placeholder variables in sql_template using var_mapping.
    Placeholders are expected in double quotes, e.g. "airport_code0".
    """
    out = sql_template
    for var, val in var_mapping.items():
        out = out.replace(f'"{var}"', f'"{val}"')
    return out


def compute_accuracy(predicted_sqls: list[str], gold_sql_lists: list[list[str]]) -> float:
    """
    Compute exact-match accuracy: a prediction is correct if it exactly matches
    any of the gold SQLs (after normalization).
    """
    correct = 0
    for pred, golds in zip(predicted_sqls, gold_sql_lists):
        p = normalize_sql(pred)
        norms = [normalize_sql(g) for g in golds]
        if p in norms:
            correct += 1
    return correct / len(predicted_sqls) if predicted_sqls else 0.0


def collate_classification(batch: list[tuple[list[int], int]], pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for classification: batch of (input_ids, template_id).
    Pads input_ids to max length with pad_idx.
    Returns: (batch_inputs, batch_labels).
    """
    inputs, labels = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in inputs]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_tagging(batch: list[tuple[list[int], list[int]]], pad_idx: int, tag_pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for tagging: batch of (input_ids, tag_ids).
    Pads both sequences to max length.
    Returns: (batch_inputs, batch_tag_labels).
    """
    inputs, tags = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    padded_inputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in inputs]
    padded_tags   = [seq + [tag_pad_idx] * (max_len - len(seq)) for seq in tags]
    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_tags, dtype=torch.long)


def collate_joint(batch: list[tuple[list[int], int, list[int]]], pad_idx: int, tag_pad_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate for joint models: batch of (input_ids, template_id, tag_ids).
    Pads sequences and returns (inputs, template_labels, tag_labels).
    """
    input_ids, tmpl_ids, tag_ids = zip(*batch)
    max_len = max(len(seq) for seq in input_ids)
    padded_inputs = [seq + [pad_idx] * (max_len - len(seq)) for seq in input_ids]
    padded_tags   = [seq + [tag_pad_idx] * (max_len - len(seq)) for seq in tag_ids]
    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(tmpl_ids, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long)
    )
