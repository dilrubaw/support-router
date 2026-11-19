import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:  # eski torch
    from torch.quantization import quantize_dynamic


def build_quantized(student_dir):
    model = AutoModelForSequenceClassification.from_pretrained(student_dir).to("cpu").eval()
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def main():
    q = build_quantized(config.ARTIFACTS / "student")
    out = config.ARTIFACTS / "student_int8.pt"
    torch.save(q.state_dict(), out)
    print(f"Kuantize model kaydedildi: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
