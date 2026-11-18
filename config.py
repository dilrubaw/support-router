"""Proje ayarları. Tüm scriptler buradan okur."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS = ROOT / "artifacts"      # model ağırlıkları (GitHub'a KONMAZ, Hugging Face Hub'a yüklenir)
RESULTS = ROOT / "results"          # sonuç tabloları (GitHub'a konur)

# >>> BUNU DEĞİŞTİR: kendi Hugging Face kullanıcı adın <<<
HF_REPO_ID = "KULLANICI_ADIN/support-router-models"

SEED = 42
OOS_LABEL = "oos"

# Transformer modelleri (ikisi de çok dilli -> Türkçe + İngilizce)
TEACHER_NAME = "bert-base-multilingual-cased"          # Öğretmen (Transformer, FP32)
STUDENT_NAME = "distilbert-base-multilingual-cased"    # Öğrenci (distillation + INT8 quantization)
MAX_LEN = 64

TEACHER_EPOCHS, TEACHER_LR = 5, 4e-5
STUDENT_EPOCHS, STUDENT_LR = 5, 5e-5
BATCH_SIZE = 32
KD_TEMPERATURE, KD_ALPHA = 2.0, 0.5

# Kendi el yazımı örneklerimiz az olduğu için eğitim setinde kaç kez tekrar edilecek
CUSTOM_OVERSAMPLE = 4

# Gecikme ölçümü (R4: >=100 çalıştırma)
LATENCY_RUNS = 100
LATENCY_WARMUP = 10
