"""Adım 1: Veriyi indir, birleştir, böl, ön işle.
CLINC150 'plus' (train/val/test + out-of-scope) + kendi yazdığımız 120 TR/EN beyaz eşya mesajı.
Çıktı: data/processed/{train,val,test}.csv, artifacts/labels.json, artifacts/vocab.json
Çalıştır: python -m src.prepare_data
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocess import Preprocessor, normalize, tokenize  # noqa: E402


def load_clinc():
    from datasets import load_dataset
    ds, last_err = None, None
    for kwargs in [dict(path="clinc/clinc_oos"), dict(path="clinc_oos"),
                   dict(path="clinc/clinc_oos", revision="refs/convert/parquet")]:
        try:
            ds = load_dataset(name="plus", **kwargs)
            break
        except Exception as e:  # farklı datasets sürümleri için yedek yollar
            print("Yükleme denemesi başarısız:", kwargs, e)
            last_err = e
    if ds is None:
        raise last_err
    names = ds["train"].features["intent"].names
    out = {}
    for split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        out[key] = pd.DataFrame({
            "text": ds[split]["text"],
            "label": [names[i] for i in ds[split]["intent"]],
            "source": "clinc", "lang": "en",
        })
    return out


def split_custom(path, seed):
    df = pd.read_csv(path).rename(columns={"intent": "label"})
    df["source"] = "custom"
    parts = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)
    for _, g in df.groupby("label"):
        g = g.iloc[rng.permutation(len(g))]
        n = len(g)
        n_tr, n_va = int(n * 0.6), int(n * 0.2)
        parts["train"].append(g.iloc[:n_tr])
        parts["val"].append(g.iloc[n_tr:n_tr + n_va])
        parts["test"].append(g.iloc[n_tr + n_va:])
    return {k: pd.concat(v) for k, v in parts.items()}


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    clinc = load_clinc()
    custom = split_custom(config.DATA_DIR / "custom_appliance.csv", config.SEED)
    splits = {
        "train": pd.concat([clinc["train"]] + [custom["train"]] * config.CUSTOM_OVERSAMPLE),
        "val": pd.concat([clinc["val"], custom["val"]]),
        "test": pd.concat([clinc["test"], custom["test"]]),
    }

    labels = sorted(splits["train"]["label"].unique())
    assert config.OOS_LABEL in labels
    label2id = {l: i for i, l in enumerate(labels)}

    # Yazım düzeltici için "bilinen kelimeler" sözlüğü sadece eğitim setinden çıkarılır
    vocab = sorted({t for txt in splits["train"]["text"] for t in tokenize(normalize(txt)[0])})
    pre = Preprocessor(vocab)

    for name, df in splits.items():
        df = df.reset_index(drop=True)
        df["clean"] = [pre.process(t)["clean"] for t in df["text"]]
        df["label_id"] = df["label"].map(label2id)
        assert df["label_id"].notna().all(), f"{name} setinde bilinmeyen etiket var"
        df.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} örnek  (custom: {(df.source == 'custom').sum()}, "
              f"oos: {(df.label == config.OOS_LABEL).sum()})")

    (config.ARTIFACTS / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))
    (config.ARTIFACTS / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    print(f"{len(labels)} sınıf, sözlük boyutu {len(vocab)}")


if __name__ == "__main__":
    main()
