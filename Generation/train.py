#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Import your data utility (to be implemented in generation/data.py)
from Generation.data import get_dataloaders

# Import models
from models.lstm_seq2seq import Encoder as LSTMEncoder, Decoder as LSTMDecoder, Seq2Seq as LSTMSeq2Seq
from models.lstm_attention import Encoder as AttnEncoder, Decoder as AttnDecoder, Seq2SeqAttention
from models.transformer_seq2seq import TransformerSeq2Seq


def parse_args():
    parser = argparse.ArgumentParser(description="Train Seq2Seq models for ATIS SQL generation")
    parser.add_argument('--model', type=str, choices=['lstm','attn','transformer'], default='lstm',
                        help='Which model to train: lstm | attn | transformer')
    parser.add_argument('--data-path', type=str, default='../data.json', help='Path to processed data file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--emb-dim', type=int, default=256)
    parser.add_argument('--hid-dim', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-heads', type=int, default=8, help='Transformer number of heads')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--tf-ratio', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, args, src_pad_idx, tgt_pad_idx):
    model.train()
    epoch_loss = 0
    for batch in loader:
        src, src_len, trg = batch.src, batch.src_len, batch.trg
        src, src_len, trg = src.to(args.device), src_len.to(args.device), trg.to(args.device)
        optimizer.zero_grad()
        # Forward pass
        if args.model == 'lstm':
            output = model(src, src_len, trg, args.tf_ratio)
        elif args.model == 'attn':
            output = model(src, src_len, trg, args.tf_ratio, src_pad_idx)
        else:  # transformer
            # prepare input/output for transformer
            trg_input = trg[:, :-1]
            src_mask = None
            tgt_mask = model.generate_square_subsequent_mask(trg_input.size(1)).to(args.device)
            src_padding_mask = (src == src_pad_idx)
            tgt_padding_mask = (trg_input == tgt_pad_idx)
            memory_key_padding_mask = src_padding_mask
            output = model(src, src_mask, trg_input, tgt_mask,
                           src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # output: [batch_size, trg_len (or trg_len-1), output_dim]
        # align output and trg for loss: skip first token
        if args.model == 'transformer':
            output = output.reshape(-1, output.shape[-1])  # skip <sos> externally
            trg_y = trg[:, 1:].reshape(-1)
        else:
            output = output[:, 1:, :].reshape(-1, output.shape[-1])
            trg_y = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            src, src_len, trg = batch.src, batch.src_len, batch.trg
            src, src_len = src.to(args.device), src_len.to(args.device)
            # generate predictions
            if args.model in ['lstm', 'attn']:
                preds = model.inference(src, src_len, tgt_sos_idx, tgt_eos_idx, max_len=trg.size(1))
            else:
                preds = model.inference(src, src_len, tgt_sos_idx, tgt_eos_idx, max_len=trg.size(1), pad_idx=src_pad_idx)
            # convert and compare sequences
            for i in range(src.size(0)):
                pred_seq = preds[i].tolist()
                trg_seq = trg[i].tolist()
                # truncate at eos
                if tgt_eos_idx in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(tgt_eos_idx)+1]
                if tgt_eos_idx in trg_seq:
                    trg_seq = trg_seq[:trg_seq.index(tgt_eos_idx)+1]
                if pred_seq == trg_seq:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # Load data
    train_loader, dev_loader, test_loader, \
        src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx, input_dim, output_dim = \
        get_dataloaders(args.data_path, args.batch_size)
    args.device = torch.device(args.device)

    # Initialize model
    if args.model == 'lstm':
        enc = LSTMEncoder(input_dim, args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
        dec = LSTMDecoder(output_dim, args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
        model = LSTMSeq2Seq(enc, dec, args.device)
    elif args.model == 'attn':
        enc = AttnEncoder(input_dim, args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
        dec = AttnDecoder(output_dim, args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
        model = Seq2SeqAttention(enc, dec, args.device)
    else:
        model = TransformerSeq2Seq(input_dim, output_dim, args.emb_dim,
                                    args.n_heads, args.hid_dim, args.n_layers,
                                    args.dropout, args.device)
    model = model.to(args.device)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_dev_acc = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args, src_pad_idx, tgt_pad_idx)
        dev_acc = evaluate(model, dev_loader, args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx)
        print(f'Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Dev Acc: {dev_acc:.2%}')
        # Save best
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model}_best.pt'))

    # Final test evaluation
    test_acc = evaluate(model, test_loader, args, src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx)
    print(f'Test Acc ({args.model}): {test_acc:.2%}')


if __name__ == '__main__':
    main()
