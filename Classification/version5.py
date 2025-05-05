# -*- coding: utf-8 -*-
"""run_atis_bert.py —— GPU 版本，输出模板分类准确率和整体准确率
=======================================================
* 全流程（加载 → 训练 → 推理）均在 GPU 上运行（如可用）
* 若无 CUDA，则自动退回 CPU，不影响正确性

依赖：torch>=2.0  transformers>=4.39
用法：
    CUDA_VISIBLE_DEVICES=0 python run_atis_bert.py
"""
import json

from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用设备: {DEVICE}")

data_path = Path(__file__).parent / "atis.json"
assert data_path.exists(), "请将 atis.json 与脚本置于同一目录"
with data_path.open(encoding="utf-8") as f:
    data = json.load(f)

# 根据 question-split 字段划分
train_entries, test_entries = [], []
for entry in data:
    train_sentences, test_sentences = [], []
    for sent in entry.get("sentences", []):
        if sent.get("question-split") in {"train", "dev"}:
            train_sentences.append(sent)
        elif sent.get("question-split") == "test":
            test_sentences.append(sent)
    if train_sentences:
        train_entries.append({"sql": entry["sql"], "variables": entry.get("variables", []), "sentences": train_sentences})
    if test_sentences:
        test_entries.append({"sql": entry["sql"], "variables": entry.get("variables", []), "sentences": test_sentences})

# 模板映射
all_templates = set()
for entry in data:
    all_templates.update(entry["sql"])
template_list = list(all_templates)
template2id = {tmpl: idx for idx, tmpl in enumerate(template_list)}
NUM_TEMPLATES = len(template_list)
print(f"模板总数: {NUM_TEMPLATES}")

# BIO 标签集
label_set = {"O"}
for e in data:
    for v in e.get("variables", []):
        name = v.get("name")
        if name:
            label_set.update({f"B-{name}", f"I-{name}"})
label2id = {lab: i for i, lab in enumerate(sorted(label_set))}
id2label = {i: lab for lab, i in label2id.items()}

def replace_ph(text: str, vars_: dict) -> str:
    for k, v in vars_.items():
        text = text.replace(k, v)
    return text

def word_tags(text: str, vars_: dict):
    words = text.split()
    tags  = ["O"] * len(words)
    for var, val in vars_.items():
        tokens = val.split()
        for i in range(len(words) - len(tokens) + 1):
            if words[i:i+len(tokens)] == tokens:
                tags[i] = f"B-{var}"
                for j in range(1, len(tokens)):
                    tags[i+j] = f"I-{var}"
    return tags

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")

def encode(texts, word_level_tags=None):
    enc = TOKENIZER(texts, truncation=True, padding=True, return_offsets_mapping=True)
    if word_level_tags is None:
        return enc, None
    aligned = []
    max_len = len(enc["input_ids"][0])
    for i, tags in enumerate(word_level_tags):
        word_ids = enc.word_ids(batch_index=i)
        seq = []
        prev = None
        for wid in word_ids:
            if wid is None:
                seq.append(-100)
            else:
                label = tags[wid] if wid < len(tags) else "O"
                if wid == prev and label.startswith("B-"):
                    label = "I-" + label[2:]
                seq.append(label2id[label])
            prev = wid
        seq += [-100] * (max_len - len(seq))
        aligned.append(seq)
    return enc, torch.tensor(aligned)

# 构造训练集
train_texts, train_temp_ids, train_tags = [], [], []
for entry in train_entries:
    tmpl = min(entry["sql"], key=len)
    tid = template2id[tmpl]
    for sent in entry.get("sentences", []):
        txt = replace_ph(sent.get("text", ""), sent.get("variables", {}))
        train_texts.append(txt)
        train_temp_ids.append(tid)
        train_tags.append(word_tags(txt, sent.get("variables", {})))

enc_cls = TOKENIZER(train_texts, truncation=True, padding=True)
cls_labels = torch.tensor(train_temp_ids)
enc_tok, tok_labels = encode(train_texts, train_tags)

class SimpleDS(Dataset):
    def __init__(self, encodings, labels):
        self.enc, self.labels = encodings, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k in ["input_ids", "attention_mask"]}
        item["labels"] = self.labels[idx]
        return item

train_ds_cls = SimpleDS(enc_cls, cls_labels)
train_ds_tok = SimpleDS(enc_tok, tok_labels)

cls_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_TEMPLATES).to(DEVICE)
tok_model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

args_cls = TrainingArguments("cls_out", num_train_epochs=15, per_device_train_batch_size=16, learning_rate=5e-5, logging_steps=1000, evaluation_strategy="epoch",
    save_strategy="epoch")
args_tok = TrainingArguments("tok_out", num_train_epochs=15, per_device_train_batch_size=16, learning_rate=3e-5, logging_steps=1000, evaluation_strategy="epoch",
    save_strategy="epoch")

print("\n>>> 训练模板分类器 ...")
Trainer(model=cls_model, args=args_cls, train_dataset=train_ds_cls, eval_dataset=train_ds_cls,).train()

print("\n>>> 训练槽位标注器 ...")
Trainer(model=tok_model, args=args_tok, train_dataset=train_ds_tok, eval_dataset=train_ds_tok, data_collator=DataCollatorForTokenClassification(TOKENIZER)).train()

# 后续测试推理代码可照旧处理 test_entries 数据集


# --------------------------------------------------
# 8. 构造测试集并推理
# --------------------------------------------------
test_texts, gold_sqls, test_tids = [], [], []
for entry in test_entries:
    tmpl = min(entry["sql"], key=len)
    tid = template2id[tmpl]
    for sent in entry.get("sentences", []):
        txt = replace_ph(sent.get("text", ""), sent.get("variables", {}))
        test_texts.append(txt)
        test_tids.append(tid)
        # 构建 SQL 金标准列表
        variants = []
        for sql in entry.get("sql", []):
            filled = sql
            for var, val in sent.get("variables", {}).items():
                filled = filled.replace(f'"{var}"', f'"{val}"')
                filled = filled.replace(var, val)
            variants.append("".join(filled.split()))
        gold_sqls.append(variants)

enc_test = TOKENIZER(test_texts, truncation=True, padding=True, return_offsets_mapping=True, return_tensors="pt").to(DEVICE)

# 推理
cls_model.eval(); tok_model.eval()
with torch.no_grad():
    logits_cls = cls_model(input_ids=enc_test["input_ids"], attention_mask=enc_test["attention_mask"]).logits
    pred_tids = logits_cls.argmax(dim=1).tolist()

    logits_tok = tok_model(input_ids=enc_test["input_ids"], attention_mask=enc_test["attention_mask"]).logits
    pred_tag_ids = logits_tok.argmax(dim=-1).tolist()

correct_tmpl = sum(int(pred == gold) for pred, gold in zip(pred_tids, test_tids))
tmpl_acc = correct_tmpl / len(test_tids)
print(f"\n>>> 模板分类准确率: {tmpl_acc:.3f} ({correct_tmpl}/{len(test_tids)})")

correct_full = 0
offsets = enc_test["offset_mapping"].cpu()
for i, txt in enumerate(test_texts):
    tid = pred_tids[i]
    if tid >= len(template_list):
        continue
    tmpl_sql = template_list[tid]
    word_ids = enc_test.word_ids(batch_index=i)
    tags = [id2label[t] if t != -100 else "O" for t in pred_tag_ids[i]]

    entities, cur = [], None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            cur = None; continue
        tag = tags[idx]
        if tag == "O":
            cur = None; continue
        bio, name = tag.split('-', 1)
        st, ed = offsets[i][idx].tolist()
        if bio == "B" or cur is None or cur[2] != name:
            cur = [st, ed, name]; entities.append(cur)
        else:
            cur[1] = ed
    pred_vals = {n: txt[s:e] for s, e, n in entities}

    pred_sql = tmpl_sql
    for var, val in pred_vals.items():
        pred_sql = pred_sql.replace(f'"{var}"', f'"{val}"')
        pred_sql = pred_sql.replace(var, val)

    if "".join(pred_sql.split()) in gold_sqls[i]:
        correct_full += 1

full_acc = correct_full / len(test_texts)
print(f"\n>>> 整体准确率: {full_acc:.3f} ({correct_full}/{len(test_texts)})")