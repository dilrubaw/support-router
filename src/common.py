"""Ortak yardımcılar: veri yükleme, eşik (threshold) uygulama."""
import json

import numpy as np
import pandas as pd

import config


def load_split(name):
    return pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", keep_default_na=False)


def load_labels():
    return json.loads((config.ARTIFACTS / "labels.json").read_text())


def oos_id():
    return load_labels().index(config.OOS_LABEL)


def apply_threshold(probs, threshold, oos_index):
    pred = probs.argmax(1).copy()
    pred[probs.max(1) < threshold] = oos_index
    return pred


def tune_threshold(probs, y, oos_index):
    """Doğrulama (val) setinde en iyi doğruluğu veren güven eşiğini seç."""
    best_t, best_acc = 0.0, -1.0
    for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
        acc = float((apply_threshold(probs, t, oos_index) == y).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc
