import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

import config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_logits(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=config.MAX_LEN, return_tensors="pt").to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def predict_proba(model, tokenizer, texts, device, batch_size=64):
    return torch.softmax(predict_logits(model, tokenizer, texts, device, batch_size), -1).numpy()


def train_classifier(model, tokenizer, texts, labels, val_texts, val_labels, out_dir,
                     epochs, lr, teacher_logits=None):
    device = get_device()
    model.to(device)
    texts, labels = list(texts), np.asarray(labels)
    n, bs = len(texts), config.BATCH_SIZE
    steps = epochs * math.ceil(n / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(config.SEED)
    T, alpha = config.KD_TEMPERATURE, config.KD_ALPHA

    best = -1.0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                            max_length=config.MAX_LEN, return_tensors="pt").to(device)
            y = torch.as_tensor(labels[idx], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            logits = logits.float()
            loss = F.cross_entropy(logits, y)
            if teacher_logits is not None:
                t = teacher_logits[torch.as_tensor(idx)].to(device)
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t / T, -1),
                              reduction="batchmean") * T * T
                loss = alpha * loss + (1 - alpha) * kd
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * len(idx)
        val_acc = float((predict_proba(model, tokenizer, val_texts, device).argmax(1)
                         == np.asarray(val_labels)).mean())
        print(f"epoch {ep + 1}/{epochs}  loss {running / n:.4f}  val acc {val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    print(f"En iyi val accuracy: {best:.4f} -> {out_dir}")
    return best
