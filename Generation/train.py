#!/usr/bin/env python3
# File: A4/Generation/train.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# 从 data.py 导入加载函数和特殊 token 常量
from data import get_dataloaders, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

# 导入三种模型
from models.lstm_seq2seq import Encoder      as LSTMEncoder, \
                                       Decoder      as LSTMDecoder, \
                                       Seq2Seq      as LSTMSeq2Seq
from models.lstm_attention import Encoder       as AttnEncoder, \
                                         Decoder       as AttnDecoder, \
                                         Seq2SeqAttention
from models.transformer_seq2seq import TransformerSeq2Seq


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / "atis.json"

    p = argparse.ArgumentParser(description="Train Seq2Seq models for ATIS SQL generation")
    p.add_argument("--models", nargs="+",
                   choices=["lstm", "attn", "transformer"],
                   default=["lstm", "attn", "transformer"],
                   help="Which model(s) to train; default: all three")
    p.add_argument("--data-path", type=str, default=str(default_data),
                   help="Path to raw atis.json (must contain 'sentences')")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--emb-dim",    type=int, default=256)
    p.add_argument("--hid-dim",    type=int, default=512)
    p.add_argument("--n-layers",   type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--tf-ratio",   type=float, default=0.5,
                   help="Teacher forcing ratio for LSTM/Attention")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--checkpoints", type=str, default="checkpoints",
                   help="Directory to save best model checkpoints")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(name, input_dim, output_dim, args):
    """实例化 lstm / attn / transformer 三种模型（移除多余 tf_ratio）"""
    if name == "lstm":
        enc = LSTMEncoder(input_dim, args.emb_dim, args.hid_dim,
                          args.n_layers, args.dropout)
        dec = LSTMDecoder(output_dim, args.emb_dim, args.hid_dim,
                          args.n_layers, args.dropout)
        return LSTMSeq2Seq(enc, dec, args.device).to(args.device)

    if name == "attn":
        enc = AttnEncoder(input_dim, args.emb_dim, args.hid_dim,
                          args.n_layers, args.dropout)
        dec = AttnDecoder(output_dim, args.emb_dim, args.hid_dim,
                          args.n_layers, args.dropout)
        return Seq2SeqAttention(enc, dec, args.device).to(args.device)

    # transformer
    model = TransformerSeq2Seq(
        input_dim, output_dim,
        args.emb_dim,           # emb_dim
        args.hid_dim,           # hid_dim used as model_dim
        args.n_layers,          # number of layers
        args.dropout,           # dropout
        args.device             # device
    )
    return model.to(args.device)


def train_epoch(model, loader, optimizer, criterion,
                args, src_pad_idx, tgt_pad_idx):
    """单轮训练，返回平均 loss"""
    model.train()
    total_loss = 0.0

    for batch in loader:
        src, src_len, trg = batch['src'], batch['src_len'], batch['trg']
        src = src.to(args.device)
        src_len = src_len.to(args.device)
        trg = trg.to(args.device)

        optimizer.zero_grad()

        # 前向
        if isinstance(model, (LSTMSeq2Seq, Seq2SeqAttention)):
            # 接口： forward(src, src_len, trg, tf_ratio)
            output = model(src, src_len, trg, args.tf_ratio)
        else:
            # Transformer 接口
            trg_input = trg[:, :-1]
            tgt_mask = model.generate_square_subsequent_mask(
                trg_input.size(1)
            ).to(args.device)
            src_pad_mask = (src == src_pad_idx)
            tgt_pad_mask = (trg_input == tgt_pad_idx)
            output = model(
                src, None,
                trg_input, tgt_mask,
                src_pad_mask, tgt_pad_mask,
                src_pad_mask
            )

        # 计算 loss：只对 t=1…T-1 步的输出计算
        # output: [B, T, V]. trg: [B, T]
        output = output[:, 1:, :]            # drop t=0 step
        B, Tm1, V = output.shape             # Tm1 = T-1
        loss = criterion(
            output.contiguous().view(B * Tm1, V),
            trg[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, args,
             src_pad_idx, tgt_pad_idx,
             tgt_sos_idx, tgt_eos_idx):
    """在指定数据集上做推理，返回 SQL 完全匹配的准确率"""
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in loader:
            src, src_len = batch['src'], batch['src_len']
            src = src.to(args.device)
            src_len = src_len.to(args.device)

            # 推理接口
            if isinstance(model, (LSTMSeq2Seq, Seq2SeqAttention)):
                preds = model.inference(
                    src, src_len, tgt_sos_idx, tgt_eos_idx
                )
            else:
                preds = model.inference(
                    src, src_pad_idx, tgt_sos_idx, tgt_eos_idx
                )

            gold_seqs = batch['trg'].tolist()
            for pred_seq, gold_seq in zip(preds, gold_seqs):
                # gold_seq 包含 <sos>…<eos>…pad
                if tgt_eos_idx in gold_seq:
                    gl = gold_seq.index(tgt_eos_idx) + 1
                else:
                    gl = len(gold_seq)
                gold_trim = gold_seq[1:gl]
                pred_trim = pred_seq[: gl - 1]
                total += 1
                if pred_trim == gold_trim:
                    correct += 1

    return correct / total if total > 0 else 0.0


def train_and_eval(model_name, loaders, pad_idxs, idxs, args):
    """对单一模型做 train→Dev 选最优→Test 测评"""
    train_loader, dev_loader, test_loader = loaders
    src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx = pad_idxs
    input_dim, output_dim = idxs

    model = build_model(model_name, input_dim, output_dim, args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_dev = 0.0
    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            args, src_pad_idx, tgt_pad_idx
        )
        dv_acc = evaluate(
            model, dev_loader, args,
            src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx
        )
        print(f"[{model_name:>10s}] Ep{epoch}/{args.epochs} "
              f"Loss={tr_loss:.4f} DevAcc={dv_acc:.2%}")

        if dv_acc > best_dev:
            best_dev = dv_acc
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{model_name}_best.pt"
            )

    # 测试集最终评估
    model.load_state_dict(torch.load(ckpt_dir / f"{model_name}_best.pt"))
    te_acc = evaluate(
        model, test_loader, args,
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx
    )
    print(f"[{model_name:>10s}] **TEST ACC={te_acc:.2%}**\n")


def main():
    args = parse_args()

    # 1) 先取三元组 loaders
    loaders_tuple, pad_idxs, idxs = get_dataloaders(
        args.data_path, args.batch_size
    )
    train_loader, dev_loader, test_loader = loaders_tuple

    # 2) pad 和 vocab 大小都在 pad_idxs, idxs 中
    src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx = pad_idxs
    input_dim, output_dim = idxs

    loaders = (train_loader, dev_loader, test_loader)
    pad_idxs = (src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx)
    idxs     = (input_dim, output_dim)

    for m in args.models:
        print(f"\n=== Start: {m} ===")
        train_and_eval(m, loaders, pad_idxs, idxs, args)


if __name__ == "__main__":
    main()
