import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.common import apply_threshold, load_labels, load_split, tune_threshold 
from src.predictors import load_predictors 


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "bilinmiyor"


def measure_latency(pred, texts):
    for t in texts[:config.LATENCY_WARMUP]:
        pred.predict_proba([t])
    times = []
    for i in range(config.LATENCY_RUNS):
        t0 = time.perf_counter()
        pred.predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    config.RESULTS.mkdir(exist_ok=True)
    labels = load_labels()
    oos = labels.index(config.OOS_LABEL)
    val, test = load_split("val"), load_split("test")
    yv, yt = val["label_id"].to_numpy(), test["label_id"].to_numpy()
    is_oos = yt == oos
    custom = (test["source"] == "custom").to_numpy()
    tr = (test["lang"] == "tr").to_numpy()
    rng = np.random.default_rng(config.SEED)
    lat_texts = list(test["clean"].iloc[rng.choice(len(test), config.LATENCY_RUNS, replace=False)])

    gpu = torch.cuda.is_available()
    rows, thresholds = [], {}
    for key, pred in load_predictors().items():
        print(f"\n=== {pred.name} ===")
        if gpu and hasattr(pred, "to"):
            pred.to("cuda")                     
        pv = pred.predict_proba(val["clean"])
        pt = pred.predict_proba(test["clean"])
        thr, _ = tune_threshold(pv, yv, oos)
        thresholds[key] = thr
        yp = apply_threshold(pt, thr, oos)
        if hasattr(pred, "to"):
            pred.to("cpu")                      
        lat_mean, lat_p95 = measure_latency(pred, lat_texts)

        row = {
            "Model": pred.name,
            "Hassasiyet": pred.precision,
            "Accuracy (tüm test)": (yp == yt).mean(),
            "In-scope accuracy": (yp[~is_oos] == yt[~is_oos]).mean(),
            "OOS recall": (yp[is_oos] == oos).mean(),
            "Beyaz eşya acc. (TR+EN)": (yp[custom] == yt[custom]).mean(),
            "Türkçe acc.": (yp[tr] == yt[tr]).mean(),
            "Eşik (val)": thr,
            "Boyut (MB)": pred.size_mb(),
            "Gecikme ort. (ms)": lat_mean,
            "Gecikme p95 (ms)": lat_p95,
        }
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

        if key == "student_int8":
            wrong = yp != yt
            pd.DataFrame({
                "text": test["text"][wrong], "clean": test["clean"][wrong],
                "true": [labels[i] for i in yt[wrong]], "pred": [labels[i] for i in yp[wrong]],
                "confidence": pt.max(1)[wrong].round(3),
            }).to_csv(config.RESULTS / "errors_student_int8.csv", index=False)
        del pred

    config.RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "results_table.csv", index=False)
    (config.ARTIFACTS / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    hw = (f"CPU: {cpu_name()}\nCPU thread sayısı (torch): {torch.get_num_threads()}\n"
          f"Platform: {platform.platform()}\nPython {platform.python_version()}, "
          f"torch {torch.__version__}\nGecikme: CPU, batch=1, {config.LATENCY_RUNS} çalıştırma "
          f"ortalaması ({config.LATENCY_WARMUP} ısınma sonrası), tokenization dahil.\n"
          f"Test seti: {len(test)} mesaj (CLINC150 test {int((~custom).sum())} "
          f"+ kendi verimiz {int(custom.sum())}), oos: {int(is_oos.sum())}\n")
    (config.RESULTS / "hardware.txt").write_text(hw, encoding="utf-8")

    fmt = df.copy()
    for c in fmt.columns:
        if "acc" in c.lower() or "recall" in c.lower():
            fmt[c] = (fmt[c] * 100).map("{:.1f}%".format)
        elif c in ("Boyut (MB)", "Gecikme ort. (ms)", "Gecikme p95 (ms)"):
            fmt[c] = fmt[c].map("{:.1f}".format)
    md = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    md += "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in fmt.values)
    (config.RESULTS / "results_table.md").write_text(md + "\n\n" + hw, encoding="utf-8")
    print("\n" + md + "\n\n" + hw)


if __name__ == "__main__":
    main()
