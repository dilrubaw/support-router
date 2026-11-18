"""word2vec (skip-gram + negative sampling) ve ileri beslemeli ağ (FFNN) tanımları."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocess import tokenize


class SkipGramNS(nn.Module):
    """word2vec skip-gram, negative sampling (Mikolov et al., 2013)."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        v = self.in_emb(center)                      # (B, D)
        u = self.out_emb(context)                    # (B, D)
        n = self.out_emb(negatives)                  # (B, K, D)
        pos = F.logsigmoid((v * u).sum(-1))
        neg = F.logsigmoid(-torch.bmm(n, v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


class FFNN(nn.Module):
    """Ortalama word2vec vektörü -> gizli katman (ReLU) -> dropout -> softmax."""

    def __init__(self, in_dim, hidden, n_labels, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x):
        return self.net(x)


def average_embeddings(texts, word2idx, vectors):
    X = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for i, t in enumerate(texts):
        ids = [word2idx[w] for w in tokenize(t) if w in word2idx]
        if ids:
            X[i] = vectors[ids].mean(0)
    return X
