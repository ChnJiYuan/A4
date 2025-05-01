# Re-running ff_models.py after environment reset

import torch
import torch.nn as nn

class FFTemplateClassifier(nn.Module):
    """
    A feedforward model for template classification.
    Input: mean-pooled token embeddings
    Architecture: Embedding → Mean → Linear → ReLU → Linear → Output
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_templates, pad_idx=0):
        super(FFTemplateClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_templates)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # (B, T, E)
        mean_emb = emb.mean(dim=1)       # (B, E)
        return self.ff(mean_emb)         # (B, num_templates)


class FFSequenceTagger(nn.Module):
    """
    A feedforward model for sequence tagging.
    Input: per-token embeddings
    Architecture: Embedding → Linear → ReLU → Linear → Tag logits (for each token)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, pad_idx=0):
        super(FFSequenceTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tags)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # (B, T, E)
        return self.ff(emb)              # (B, T, num_tags)

