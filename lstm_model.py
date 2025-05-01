import torch
import torch.nn as nn

class LSTMClassifierTagger(nn.Module):
    """
    A single LSTM model that jointly:
    - Classifies the input question into a SQL template (based on final hidden state)
    - Predicts a tag for each word (based on output sequence)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_templates, num_tags, pad_idx=0):
        super(LSTMClassifierTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Output layers
        self.template_classifier = nn.Linear(hidden_dim, num_templates)  # From final hidden state
        self.token_tagger = nn.Linear(hidden_dim, num_tags)              # From sequence outputs

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        Returns:
            template_logits: (batch_size, num_templates)
            tag_logits: (batch_size, seq_len, num_tags)
        """
        embedded = self.embedding(input_ids)  # (B, T, E)
        outputs, (h_n, _) = self.lstm(embedded)  # outputs: (B, T, H), h_n: (1, B, H)

        # Template classification from final hidden state
        final_hidden = h_n.squeeze(0)  # (B, H)
        template_logits = self.template_classifier(final_hidden)  # (B, num_templates)

        # Sequence tagging from all outputs
        tag_logits = self.token_tagger(outputs)  # (B, T, num_tags)

        return template_logits, tag_logits
