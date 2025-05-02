import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        # Using dot-product attention, no parameters
        self.hid_dim = hid_dim

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        # mask: [batch_size, src_len]
        # compute scores
        # hidden unsqueeze -> [batch_size, 1, hid_dim]
        scores = torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2))  # [batch, 1, src_len]
        scores = scores.squeeze(1)  # [batch, src_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # attention weights
        attn_weights = F.softmax(scores, dim=1)  # [batch, src_len]
        # compute context
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hid_dim]
        context = context.squeeze(1)  # [batch, hid_dim]
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # encoder_outputs: [batch, src_len, hid_dim]
        return encoder_outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        # input: [batch]
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        # get last hidden state for attention: hidden[-1]: [batch, hid_dim]
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        # context: [batch, hid_dim]
        context = context.unsqueeze(1)  # [batch, 1, hid_dim]

        # rnn input: concat embedded and context
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch,1, emb+hid]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [batch,1,hid_dim]

        output = output.squeeze(1)  # [batch, hid_dim]
        context = context.squeeze(1)  # [batch, hid_dim]
        pred_input = torch.cat((output, context), dim=1)  # [batch, hid_dim*2]
        prediction = self.fc_out(pred_input)  # [batch, output_dim]

        return prediction, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src, src_pad_idx):
        # mask pad tokens
        mask = (src != src_pad_idx)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5, src_pad_idx=None):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        mask = self.create_mask(src, src_pad_idx) if src_pad_idx is not None else None

        input = trg[:, 0]
        for t in range(1, trg_len):
            pred, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = pred
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    def inference(self, src, src_len, sos_idx, eos_idx, max_len=100, src_pad_idx=None):
        batch_size = src.shape[0]
        outputs = []
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        mask = self.create_mask(src, src_pad_idx) if src_pad_idx is not None else None

        input = torch.LongTensor([sos_idx] * batch_size).to(self.device)
        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            top1 = pred.argmax(1)
            outputs.append(top1.unsqueeze(1))
            input = top1
        outputs = torch.cat(outputs, dim=1)
        return outputs
