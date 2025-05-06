# -*- coding: utf-8 -*-
"""optimized_atis_bert.py —— 优化版 GPU 训练代码，提高模板分类和实体识别准确率
=======================================================
* 增加学习率调度策略
* 使用warmup和weight decay优化训练
* 添加早停机制避免过拟合
* 增加模型变种选项

依赖：torch>=2.0  transformers>=4.39
用法：
    CUDA_VISIBLE_DEVICES=0 python optimized_atis_bert.py
"""
import json
import random
import argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    get_scheduler
)
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    """
    设置所有随机数生成器的种子，确保结果可复现
    这会影响以下随机过程:
    1. random模块 - 用于train_test_split中的随机抽样
    2. numpy随机函数 - 用于数据增强和打乱
    3. PyTorch随机数生成 - 模型初始化和数据加载器
    """
    random.seed(seed)  # 设置Python random模块种子
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
        # 确保CUDA操作的确定性 (可能会影响性能)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 使用命令行参数中的随机种子
set_seed(5046)

# 参数设置 - 针对RTX 4060笔记本GPU (8GB VRAM)优化
parser = argparse.ArgumentParser(description='优化版ATIS-BERT训练器')
parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                    help='预训练模型名称 (bert-base-uncased或prajjwal1/bert-small)')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
parser.add_argument('--cls_lr', type=float, default=5e-5, help='分类器学习率')
parser.add_argument('--ner_lr', type=float, default=3e-5, help='NER学习率')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='预热比例')
parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                    help='梯度累积步数，可模拟更大批量')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--fp16', action='store_true', help='使用半精度训练')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用设备: {DEVICE}")
print(f">>> 使用预训练模型: {args.model_name}")

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

# 加载适合模型的分词器
TOKENIZER = AutoTokenizer.from_pretrained(args.model_name)

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

# 分割训练集和验证集 - 使用相同的随机种子确保可复现性
# 这里使用了random模块的随机抽样功能
train_texts, val_texts, train_temp_ids, val_temp_ids, train_tags, val_tags = train_test_split(
    train_texts, train_temp_ids, train_tags, test_size=0.1, random_state=args.seed
)

# 编码训练集
enc_train_cls = TOKENIZER(train_texts, truncation=True, padding=True)
cls_train_labels = torch.tensor(train_temp_ids)
enc_train_tok, tok_train_labels = encode(train_texts, train_tags)

# 编码验证集
enc_val_cls = TOKENIZER(val_texts, truncation=True, padding=True)
cls_val_labels = torch.tensor(val_temp_ids)
enc_val_tok, tok_val_labels = encode(val_texts, val_tags)

class SimpleDS(Dataset):
    def __init__(self, encodings, labels):
        self.enc, self.labels = encodings, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items() if k in ["input_ids", "attention_mask"]}
        item["labels"] = self.labels[idx]
        return item

train_ds_cls = SimpleDS(enc_train_cls, cls_train_labels)
val_ds_cls = SimpleDS(enc_val_cls, cls_val_labels)
train_ds_tok = SimpleDS(enc_train_tok, tok_train_labels)
val_ds_tok = SimpleDS(enc_val_tok, tok_val_labels)

# 创建模型
cls_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=NUM_TEMPLATES
).to(DEVICE)

tok_model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

# 训练参数设置 - 使用更优化的配置
cls_output_dir = f"cls_out_{args.model_name.split('/')[-1]}"
tok_output_dir = f"tok_out_{args.model_name.split('/')[-1]}"

# 优化TrainingArguments配置以适应RTX 4060显存限制
# 降低批量大小、启用梯度累积和混合精度训练
args_cls = TrainingArguments(
    cls_output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积，节省显存
    learning_rate=args.cls_lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=50,  # 减少日志频率
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # 只保存最近的2个检查点，节省磁盘空间
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=args.fp16,  # 启用混合精度训练
    dataloader_num_workers=2,  # 增加数据加载线程
    seed=args.seed,  # 设置随机种子
    report_to="none",  # 禁用Wandb等报告
)

args_tok = TrainingArguments(
    tok_output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 梯度累积，节省显存
    learning_rate=args.ner_lr,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    logging_steps=50,  # 减少日志频率
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # 只保存最近的2个检查点，节省磁盘空间
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=args.fp16,  # 启用混合精度训练
    dataloader_num_workers=2,  # 增加数据加载线程
    seed=args.seed,  # 设置随机种子
    report_to="none",  # 禁用Wandb等报告
)

# 内存监控函数 - 帮助监控显存使用情况
def print_gpu_memory_usage():
    """打印当前GPU显存使用情况，帮助分析是否会OOM"""
    if torch.cuda.is_available():
        print(f"\n当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"分配显存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"缓存显存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"显存利用率: {torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100:.2f}%\n")

# 优化显存使用
# 当不使用时释放缓存，减少峰值显存
torch.cuda.empty_cache()
print_gpu_memory_usage()

# 使用早停机制
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.patience)

print("\n>>> 训练模板分类器 ...")
cls_trainer = Trainer(
    model=cls_model,
    args=args_cls,
    train_dataset=train_ds_cls,
    eval_dataset=val_ds_cls,
    callbacks=[early_stopping_callback],
)
cls_trainer.train()

print("\n>>> 训练槽位标注器 ...")
tok_trainer = Trainer(
    model=tok_model,
    args=args_tok,
    train_dataset=train_ds_tok,
    eval_dataset=val_ds_tok,
    data_collator=DataCollatorForTokenClassification(TOKENIZER),
    callbacks=[early_stopping_callback],
)
tok_trainer.train()

# --------------------------------------------------
# 评估模型性能
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
cls_model.eval()
tok_model.eval()
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