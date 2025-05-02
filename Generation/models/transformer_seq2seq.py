import torch
import torch.nn as nn
import math

def get_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask  # (sz, sz)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, emb_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, n_heads, hid_dim, n_layers, dropout, device, max_len=100):
        super(TransformerSeq2Seq, self).__init__()
        self.device = device
        self.src_tok_emb = nn.Embedding(input_dim, emb_dim)
        self.tgt_tok_emb = nn.Embedding(output_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)
        self.pos_decoder = PositionalEncoding(emb_dim, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hid_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, src, src_mask, tgt, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        src_emb = self.pos_encoder(self.src_tok_emb(src))  # (batch, src_len, emb_dim)
        tgt_emb = self.pos_decoder(self.tgt_tok_emb(tgt))  # (batch, tgt_len, emb_dim)
        memory = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        output = self.fc_out(memory)  # (batch, tgt_len, output_dim)
        return output

    def generate_square_subsequent_mask(self, sz):
        return get_subsequent_mask(sz).to(self.device)

    def inference(self, src, src_len, sos_idx, eos_idx, max_len=100, pad_idx=None):
        batch_size = src.size(0)
        device = self.device
        # create masks
        src_padding_mask = (src == pad_idx)
        src_mask = None
        # encoder output via transformer
        src_emb = self.pos_encoder(self.src_tok_emb(src))
        memory = self.transformer.encoder(
            src_emb.transpose(0,1), src_key_padding_mask=src_padding_mask
        ).transpose(0,1)
        # prepare tgt input
        ys = torch.ones(batch_size,1).fill_(sos_idx).long().to(device)
        for i in range(max_len-1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1))
            tgt_padding_mask = (ys == pad_idx)
            out = self.transformer.decoder(
                self.pos_decoder(self.tgt_tok_emb(ys)).transpose(0,1),
                memory.transpose(0,1),
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            ).transpose(0,1)
            prob = self.fc_out(out[:, -1])  # (batch, output_dim)
            next_word = prob.argmax(dim=1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if (next_word == eos_idx).all():
                break
        return ys
