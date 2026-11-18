"""Müşteri Destek Mesajı Yönlendirici – Gradio uygulaması.
Ana karar modeli: kuantize (INT8) DistilmBERT öğrenci. Diğer 4 model yan yana gösterilir.
Çalıştır: python app.py          (Colab'da otomatik public link verir)
"""
import json
import os
import sys
import time

import gradio as gr
import pandas as pd

import config
from src.eliza import eliza_reply
from src.preprocess import Preprocessor
from src.teams import SMALLTALK_TEAM, route

MAIN_MODEL = "student_int8"

if not (config.ARTIFACTS / "student_int8.pt").exists():
    print("Model dosyaları bulunamadı, Hugging Face Hub'dan indiriliyor...")
    from download_models import download
    download()

from src.predictors import load_predictors  # noqa: E402

LABELS = json.loads((config.ARTIFACTS / "labels.json").read_text())
THRESHOLDS = json.loads((config.ARTIFACTS / "thresholds.json").read_text())
PRE = Preprocessor(json.loads((config.ARTIFACTS / "vocab.json").read_text()))
PREDICTORS = load_predictors()
print("Tüm modeller yüklendi (CPU).")


def analyze(message):
    if not message or not message.strip():
        return "Lütfen bir mesaj yazın.", pd.DataFrame()
    p = PRE.process(message)

    rows, main = [], None
    for key, pred in PREDICTORS.items():
        t0 = time.perf_counter()
        probs = pred.predict_proba([p["clean"]])[0]
        ms = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())
        intent, conf = LABELS[idx], float(probs[idx])
        team = route(intent, conf, THRESHOLDS[key])
        rows.append({"Model": pred.name, "Niyet (intent)": intent, "Güven": round(conf, 3),
                     "Yönlendirilen ekip": team, "Süre (ms)": round(ms, 1)})
        if key == MAIN_MODEL:
            main = (intent, conf, team, ms)

    intent, conf, team, ms = main
    reply = eliza_reply(p["normalized"], p["has_domain_signal"])
    if reply:
        team = SMALLTALK_TEAM

    corr = ", ".join(f"{a} → {b}" for a, b in p["corrections"]) or "—"
    codes = ", ".join(p["model_codes"]) or "—"
    md = f"""## ➜ Yönlendirilen ekip: **{team}**
**Ana model:** DistilmBERT öğrenci, **INT8 kuantize** (CPU) · niyet `{intent}` · güven **{conf:.2f}** (eşik {THRESHOLDS[MAIN_MODEL]:.2f}) · {ms:.1f} ms

| Ön işleme adımı | Sonuç |
|---|---|
| Normalize metin (regex) | `{p['normalized']}` |
| Çıkarılan model kodu | {codes} |
| Yazım düzeltme (edit distance) | {corr} |
| Modele giden metin | `{p['clean']}` |
"""
    if reply:
        md += f"\n**🤖 Kural tabanlı (ELIZA) cevap:** {reply}\n"
    elif team.startswith("İnsan"):
        md += "\n⚠️ Güven düşük veya kapsam dışı (out-of-scope) → mesaj bir insan temsilciye aktarıldı.\n"
    return md, pd.DataFrame(rows)


EXAMPLES = [
    "bulaşık makinem su boşaltmıyor, model DW-6534",
    "buzdolbım soğutmuyorrr!!!",
    "I lost my warranty certificate, what should I do?",
    "çamaşır makinesi kapak contası satıyor musunuz",
    "when will the technician come to install my new oven",
    "what is my bank account balance",
    "can you recommend a good movie for tonight",
    "merhaba",
    "i feel frustrated today",
    "teşekkürler",
]

with gr.Blocks(title="Destek Mesajı Yönlendirici") as demo:
    gr.Markdown("# 🧺 Beyaz Eşya Müşteri Destek Yönlendirici (TR / EN)\n"
                "Mesajı yazın → normalize edilir → niyet tahmin edilir → doğru ekibe yönlendirilir.")
    with gr.Row():
        inp = gr.Textbox(label="Müşteri mesajı", lines=2, placeholder="ör. fırınım ısınmıyor")
    btn = gr.Button("Yönlendir", variant="primary")
    out_md = gr.Markdown()
    gr.Markdown("### 5 modelin tahmini yan yana")
    out_df = gr.Dataframe(wrap=True)
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(analyze, inputs=inp, outputs=[out_md, out_df])
    inp.submit(analyze, inputs=inp, outputs=[out_md, out_df])

if __name__ == "__main__":
    share = "--share" in sys.argv or bool(os.environ.get("COLAB_RELEASE_TAG"))
    demo.launch(share=share)
