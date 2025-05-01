# Re-running transformer_model.py after kernel reset

import torch
import torch.nn as nn


class TransformerClassifierTagger(nn.Module):
    """
    A single Transformer encoder model that jointly:
    - Predicts SQL template ID from the [CLS] token (first position)
    - Predicts variable tag for each input token
    """

    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_templates, num_tags, pad_idx=0,
                 max_len=128):
        super(TransformerClassifierTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.template_classifier = nn.Linear(embedding_dim, num_templates)
        self.token_tagger = nn.Linear(embedding_dim, num_tags)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        Returns:
            template_logits: (batch_size, num_templates)
            tag_logits: (batch_size, seq_len, num_tags)
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        token_emb = self.embedding(input_ids)  # (B, T, E)
        pos_emb = self.position_embedding(positions)  # (B, T, E)
        x = token_emb + pos_emb  # (B, T, E)

        encoded = self.transformer_encoder(x)  # (B, T, E)
        cls_output = encoded[:, 0, :]  # Use first token for classification

        template_logits = self.template_classifier(cls_output)  # (B, num_templates)
        tag_logits = self.token_tagger(encoded)  # (B, T, num_tags)

        return template_logits, tag_logits

