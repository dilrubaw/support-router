import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  
from src.common import load_labels, load_split  
from src.hf_utils import train_classifier 


def main():
    torch.manual_seed(config.SEED)
    labels = load_labels()
    train, val = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(config.TEACHER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TEACHER_NAME, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    train_classifier(model, tok, train["clean"], train["label_id"], val["clean"], val["label_id"],
                     config.ARTIFACTS / "teacher", config.TEACHER_EPOCHS, config.TEACHER_LR)


if __name__ == "__main__":
    main()
