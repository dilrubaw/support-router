import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.common import load_labels, load_split  
from src.ffnn_model import FFNN, average_embeddings  

HIDDEN, DROPOUT, EPOCHS, BATCH, LR, MOMENTUM = 256, 0.3, 60, 64, 0.05, 0.9


def main():
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    w2v = torch.load(config.ARTIFACTS / "word2vec.pt", weights_only=False)
    w2i = {w: i for i, w in enumerate(w2v["vocab"])}
    vectors = w2v["vectors"]
    n_labels = len(load_labels())

    train, val = load_split("train"), load_split("val")
    Xtr = average_embeddings(train["clean"], w2i, vectors)
    Xva = average_embeddings(val["clean"], w2i, vectors)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6           
    Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std
    ytr = train["label_id"].to_numpy()
    yva = val["label_id"].to_numpy()

    model = FFNN(vectors.shape[1], HIDDEN, n_labels, DROPOUT)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=1e-4)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)

    best_acc, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), BATCH):
            idx = torch.from_numpy(perm[i:i + BATCH])
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()        
            opt.step()             
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1).numpy() == yva).mean()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1}/{EPOCHS}  loss {loss.item():.4f}  val acc {acc:.4f}")

    print(f"En iyi val accuracy (eşiksiz): {best_acc:.4f}")
    torch.save({
        "vocab": w2v["vocab"], "vectors": vectors, "mean": mean, "std": std,
        "state_dict": best_state, "hidden": HIDDEN, "dropout": DROPOUT, "n_labels": n_labels,
    }, config.ARTIFACTS / "ffnn.pt")


if __name__ == "__main__":
    main()
