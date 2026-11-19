import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.common import load_labels, load_split  
from src.hf_utils import get_device, predict_logits, train_classifier 


def main():
    torch.manual_seed(config.SEED)
    labels = load_labels()
    train, val = load_split("train"), load_split("val")
    device = get_device()

    t_dir = config.ARTIFACTS / "teacher"
    t_tok = AutoTokenizer.from_pretrained(t_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(t_dir).to(device)
    teacher_logits = predict_logits(teacher, t_tok, train["clean"], device).clone()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    s_tok = AutoTokenizer.from_pretrained(config.STUDENT_NAME)
    student = AutoModelForSequenceClassification.from_pretrained(
        config.STUDENT_NAME, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    train_classifier(student, s_tok, train["clean"], train["label_id"], val["clean"],
                     val["label_id"], config.ARTIFACTS / "student", config.STUDENT_EPOCHS,
                     config.STUDENT_LR, teacher_logits=teacher_logits)


if __name__ == "__main__":
    main()
