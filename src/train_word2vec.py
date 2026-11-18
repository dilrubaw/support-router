"""Kendi word2vec'imizi eğitim mesajları üzerinde eğit (Alan B).
Çalıştır: python -m src.train_word2vec
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.common import load_split  # noqa: E402
from src.ffnn_model import SkipGramNS  # noqa: E402
from src.preprocess import tokenize  # noqa: E402

DIM, WINDOW, NEG, EPOCHS, BATCH, LR, MIN_COUNT = 100, 3, 5, 10, 1024, 0.003, 2


def main():
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sents = [tokenize(t) for t in load_split("train")["clean"].drop_duplicates()]
    counts = Counter(w for s in sents for w in s)
    vocab = [w for w, c in counts.most_common() if c >= MIN_COUNT]
    w2i = {w: i for i, w in enumerate(vocab)}

    pairs = []
    for s in sents:
        ids = [w2i[w] for w in s if w in w2i]
        for i, c in enumerate(ids):
            for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
                if j != i:
                    pairs.append((c, ids[j]))
    pairs = np.array(pairs, dtype=np.int64)
    print(f"Sözlük: {len(vocab)} kelime, {len(pairs)} (merkez, bağlam) çifti")

    freq = np.array([counts[w] for w in vocab], dtype=np.float64) ** 0.75
    neg_dist = torch.tensor(freq / freq.sum(), dtype=torch.float, device=device)

    model = SkipGramNS(len(vocab), DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = rng.permutation(len(pairs))
        total = 0.0
        for i in range(0, len(pairs), BATCH):
            b = torch.from_numpy(pairs[perm[i:i + BATCH]]).to(device)
            negs = torch.multinomial(neg_dist, len(b) * NEG, replacement=True).view(len(b), NEG)
            loss = model(b[:, 0], b[:, 1], negs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"epoch {ep + 1}/{EPOCHS}  loss {total / len(pairs):.4f}")

    vectors = model.in_emb.weight.detach().cpu().numpy()
    torch.save({"vocab": vocab, "vectors": vectors}, config.ARTIFACTS / "word2vec.pt")

    # Kosinüs benzerliği ile en yakın komşular (sunumda gösterebilirsin)
    norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    lines = []
    for w in ["balance", "flight", "dishwasher", "warranty", "bulaşık", "garanti"]:
        if w in w2i:
            sims = norm @ norm[w2i[w]]
            nn_ = [vocab[j] for j in sims.argsort()[::-1][1:6]]
            lines.append(f"{w}: {', '.join(nn_)}")
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / "word2vec_neighbors.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
