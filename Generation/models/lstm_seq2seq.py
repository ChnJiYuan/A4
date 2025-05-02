import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        # Pack sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed)
        # outputs unused in basic Seq2Seq
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size] (current token)
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.rnn.hidden_size == decoder.rnn.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.rnn.num_layers == decoder.rnn.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # src_len: [batch_size]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.fc_out.out_features

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        # encode
        hidden, cell = self.encoder(src, src_len)

        # first input to decoder is <sos> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            # decode one token
            pred, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = pred
            # decide if we use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def inference(self, src, src_len, sos_idx, eos_idx, max_len=100):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        output_dim = self.decoder.fc_out.out_features

        outputs = []
        hidden, cell = self.encoder(src, src_len)

        # first input tokens: <sos>
        input = torch.LongTensor([sos_idx] * batch_size).to(self.device)

        for _ in range(max_len):
            pred, hidden, cell = self.decoder(input, hidden, cell)
            top1 = pred.argmax(1)
            outputs.append(top1.unsqueeze(1))
            input = top1

        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len]
        # optionally, truncate at eos
        return outputs
