"""5 modeli aynı arayüzle yükler: predict_proba(temiz_metin_listesi) -> (n, sınıf) olasılıklar."""
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from src.ffnn_model import FFNN, average_embeddings
from src.hf_utils import predict_proba as hf_predict_proba
from src.quantize import build_quantized


class TfidfPredictor:
    key, name = "tfidf", "1) TF-IDF + LogReg (klasik baseline)"
    precision = "FP64 (sklearn)"

    def __init__(self):
        self.path = config.ARTIFACTS / "tfidf_logreg.joblib"
        self.pipe = joblib.load(self.path)

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))

    def size_mb(self):
        return os.path.getsize(self.path) / 1e6


class FFNNPredictor:
    key, name = "ffnn", "2) word2vec + FFNN (SGD)"
    precision = "FP32"

    def __init__(self):
        self.path = config.ARTIFACTS / "ffnn.pt"
        ck = torch.load(self.path, weights_only=False)
        self.w2i = {w: i for i, w in enumerate(ck["vocab"])}
        self.vectors, self.mean, self.std = ck["vectors"], ck["mean"], ck["std"]
        self.model = FFNN(self.vectors.shape[1], ck["hidden"], ck["n_labels"], ck["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, texts):
        X = (average_embeddings(list(texts), self.w2i, self.vectors) - self.mean) / self.std
        return torch.softmax(self.model(torch.from_numpy(X.astype(np.float32))), -1).numpy()

    def size_mb(self):
        return os.path.getsize(self.path) / 1e6


class HFPredictor:
    def __init__(self, key, name, model_dir, quantized=False):
        self.key, self.name, self.quantized = key, name, quantized
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.device = torch.device("cpu")
        if quantized:
            self.weights = config.ARTIFACTS / "student_int8.pt"
            self.model = build_quantized(self.model_dir)
            self.model.load_state_dict(torch.load(self.weights, weights_only=False))
            self.precision = "INT8 (dynamic quantization)"
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            files = list(self.model_dir.glob("*.safetensors")) or list(self.model_dir.glob("*.bin"))
            self.weights = files[0]
            self.precision = "FP32"
        self.model.eval()

    def to(self, device):
        if not self.quantized:  # kuantize model sadece CPU'da çalışır
            self.device = torch.device(device)
            self.model.to(self.device)
        return self

    def predict_proba(self, texts, batch_size=64):
        return hf_predict_proba(self.model, self.tokenizer, list(texts), self.device, batch_size)

    def size_mb(self):
        return os.path.getsize(self.weights) / 1e6


def load_predictors(keys=None):
    a = config.ARTIFACTS
    makers = {
        "tfidf": TfidfPredictor,
        "ffnn": FFNNPredictor,
        "teacher": lambda: HFPredictor("teacher", "3) mBERT fine-tuned (Transformer, FP32)",
                                       a / "teacher"),
        "student": lambda: HFPredictor("student", "4a) DistilmBERT öğrenci (distilled, FP32)",
                                       a / "student"),
        "student_int8": lambda: HFPredictor("student_int8",
                                            "4b) DistilmBERT öğrenci INT8 (quantized) ★",
                                            a / "student", quantized=True),
    }
    keys = keys or list(makers)
    return {k: makers[k]() for k in keys}
