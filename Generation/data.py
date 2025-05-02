import json
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

class Vocab:
    def __init__(self, min_freq=1):
        self.token2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self.freqs = Counter()
        self.min_freq = min_freq

    def build(self, sequences):
        # sequences: list of list of tokens
        for seq in sequences:
            self.freqs.update(seq)
        for token, freq in self.freqs.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def __len__(self):
        return len(self.token2idx)

    def encode(self, seq):
        # seq: list of tokens
        return [self.token2idx.get(tok, self.token2idx[UNK_TOKEN]) for tok in seq]

    def decode(self, indices):
        return [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]


def tokenize(text):
    # simple whitespace tokenizer; can be improved to handle punctuation
    return text.strip().split()

class SQLDataset(Dataset):
    def __init__(self, examples, input_vocab, output_vocab):
        self.examples = examples
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.data = []
        for ex in self.examples:
            # tokenize
            src_tokens = tokenize(ex['question'])
            tgt_tokens = tokenize(ex['sql'])
            # add <sos> and <eos>
            tgt_tokens = [SOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
            # encode
            src_ids = input_vocab.encode(src_tokens)
            tgt_ids = output_vocab.encode(tgt_tokens)
            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    # batch: list of (src_tensor, tgt_tensor)
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    # pad
    padded_src = torch.zeros(len(batch), max_src, dtype=torch.long)
    padded_tgt = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        padded_src[i, :len(s)] = s
        padded_tgt[i, :len(t)] = t
    return {
        'src': padded_src,
        'src_len': torch.tensor(src_lens, dtype=torch.long),
        'trg': padded_tgt
    }


def get_dataloaders(data_path, batch_size, min_freq=1, splits=('train','dev','test')):
    # load data.json
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # flatten examples
    examples = []
    for item in raw:
        for sent in item['sentences']:
            question = sent['text']
            # pick shortest SQL from item['sql'] list
            # assume item['sql'] exists
            sqls = item['sql']
            # choose shortest by length, tie-breaker lexicographically
            best_sql = sorted(sqls, key=lambda x: (len(x), x))[0]
            # substitute variables
            sql = best_sql
            for var, val in sent['variables'].items():
                sql = sql.replace(var, val)
            examples.append({
                'question': question,
                'sql': sql,
                'split': sent['question-split']  # using question-split for splits
            })
    # build vocabs on train
    train_exs = [ex for ex in examples if ex['split'] == 'train']
    input_seqs = [tokenize(ex['question']) for ex in train_exs]
    output_seqs = [[SOS_TOKEN] + tokenize(ex['sql']) + [EOS_TOKEN] for ex in train_exs]
    input_vocab = Vocab(min_freq)
    output_vocab = Vocab(min_freq)
    input_vocab.build(input_seqs)
    output_vocab.build(output_seqs)
    # create datasets
    datasets = {}
    for split in splits:
        split_exs = [ex for ex in examples if ex['split'] == split]
        datasets[split] = SQLDataset(split_exs, input_vocab, output_vocab)
    # dataloaders
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(datasets['dev'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # return loaders and vocab indices
    src_pad_idx = input_vocab.token2idx[PAD_TOKEN]
    tgt_pad_idx = output_vocab.token2idx[PAD_TOKEN]
    tgt_sos_idx = output_vocab.token2idx[SOS_TOKEN]
    tgt_eos_idx = output_vocab.token2idx[EOS_TOKEN]
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)


    return (
        (train_loader, dev_loader, test_loader),  # loaders
        (src_pad_idx, tgt_pad_idx, tgt_sos_idx, tgt_eos_idx),  # pad_idxs
        (input_dim, output_dim)  # idxs
    )

