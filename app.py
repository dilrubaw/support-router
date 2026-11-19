import json
import os
import sys
import time

import gradio as gr
import pandas as pd

import config
from src.eliza import eliza_reply
from src.preprocess import Preprocessor
from src.teams import HUMAN_TEAM, SMALLTALK_TEAM, route

MAIN_MODEL = "student_int8"

if not (config.ARTIFACTS / "student_int8.pt").exists():
    print("Model files not found, downloading from Hugging Face Hub...")
    from download_models import download
    download()

from src.predictors import load_predictors  # noqa: E402

LABELS = json.loads((config.ARTIFACTS / "labels.json").read_text())
THRESHOLDS = json.loads((config.ARTIFACTS / "thresholds.json").read_text())
PRE = Preprocessor(json.loads((config.ARTIFACTS / "vocab.json").read_text()))
PREDICTORS = load_predictors()
print("All models loaded (CPU).")


def analyze(message):
    if not message or not message.strip():
        return "Please type a message.", pd.DataFrame()
    p = PRE.process(message)

    rows, main = [], None
    for key, pred in PREDICTORS.items():
        t0 = time.perf_counter()
        probs = pred.predict_proba([p["clean"]])[0]
        ms = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())
        intent, conf = LABELS[idx], float(probs[idx])
        team = route(intent, conf, THRESHOLDS[key])
        rows.append({"Model": pred.name, "Intent": intent, "Confidence": round(conf, 3),
                     "Routed team": team, "Time (ms)": round(ms, 1)})
        if key == MAIN_MODEL:
            main = (intent, conf, team, ms)

    intent, conf, team, ms = main
    reply = eliza_reply(p["normalized"], p["has_domain_signal"])
    if reply:
        team = SMALLTALK_TEAM

    corr = ", ".join(f"{a} → {b}" for a, b in p["corrections"]) or "—"
    codes = ", ".join(p["model_codes"]) or "—"
    md = f"""## ➜ Routed to: **{team}**
**Main model:** DistilmBERT student, **INT8 quantized** (CPU) · intent `{intent}` · confidence **{conf:.2f}** (threshold {THRESHOLDS[MAIN_MODEL]:.2f}) · {ms:.1f} ms

| Preprocessing step | Result |
|---|---|
| Normalized text (regex) | `{p['normalized']}` |
| Extracted model code | {codes} |
| Spelling correction (edit distance) | {corr} |
| Text sent to the models | `{p['clean']}` |
"""
    if reply:
        md += f"\n** Rule-based (ELIZA) reply:** {reply}\n"
    elif team == HUMAN_TEAM:
        md += "\n Low confidence or out-of-scope → the message is handed over to a human agent.\n"
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
    "thank you",
]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Home Appliance Customer Support Router (TR / EN)\n"
                "Type a message → it is normalized → the intent is predicted → "
                "it is routed to the right team.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message", lines=2,
                         placeholder="e.g. my oven is not heating / fırınım ısınmıyor")
    btn = gr.Button("Route", variant="primary")
    out_md = gr.Markdown()
    gr.Markdown("### Predictions of all 5 models side by side")
    out_df = gr.Dataframe(wrap=True)
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(analyze, inputs=inp, outputs=[out_md, out_df])
    inp.submit(analyze, inputs=inp, outputs=[out_md, out_df])

if __name__ == "__main__":
    share = "--share" in sys.argv or bool(os.environ.get("COLAB_RELEASE_TAG"))
    demo.launch(share=share)
