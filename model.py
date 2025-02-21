# model.py
import torch
import torch.nn as nn

class CustomRNN(nn.Module):
    def __init__(self, vocab_emb, hidden_dim):
        super(CustomRNN, self).__init__()

        vocab_size, embedding_dim = vocab_emb.shape
        self.embedding_matrix = vocab_emb.clone().detach()

        # Define RNN weights
        self.Wxh = nn.Parameter(torch.randn(embedding_dim, hidden_dim) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))
        self.Why = nn.Parameter(torch.randn(vocab_size, hidden_dim) * 0.01)
        self.by = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, X):
        batch_size, seq_length = X.shape
        batch_embeddings = self.embedding_matrix[X]

        ht = torch.zeros(batch_size, self.Whh.shape[0], device=X.device)
        outputs = []

        for t in range(seq_length):
            Xt = batch_embeddings[:, t, :]
            ht = torch.tanh((Xt @ self.Wxh) + (ht @ self.Whh) + self.bh)
            yt = ht @ self.Why.T + self.by
            outputs.append(yt)

        return torch.stack(outputs, dim=1)