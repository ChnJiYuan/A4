# linear_models.py
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.polys.rootisolation import A4
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import re
import random


class LinearTemplateClassifier(nn.Module):
    """
    A linear classifier for predicting the SQL template ID given a sentence.
    This uses a bag-of-words embedding averaged over tokens, passed to a linear layer.
    """
    def __init__(self, vocab_size, embedding_dim, num_templates):
        super(LinearTemplateClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_templates)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        Returns: (batch_size, num_templates)
        """
        # Embed and average over time (simple bag-of-words)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, emb_dim)
        mean_emb = embedded.mean(dim=1)  # (batch_size, emb_dim)
        logits = self.linear(mean_emb)   # (batch_size, num_templates)
        return logits


class LinearSequenceTagger(nn.Module):
    """
    A linear tagger for variable detection — predicts a label per word.
    """
    def __init__(self, vocab_size, embedding_dim, num_tags):
        super(LinearSequenceTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_tags)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        Returns: (batch_size, seq_len, num_tags)
        """
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, emb_dim)
        logits = self.linear(embedded)        # (batch_size, seq_len, num_tags)
        return logits

