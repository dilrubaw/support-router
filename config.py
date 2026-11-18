from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS = ROOT / "artifacts"      
RESULTS = ROOT / "results"         

HF_REPO_ID = "dilrubaw/support-router-models"

SEED = 42
OOS_LABEL = "oos"
TEACHER_NAME = "bert-base-multilingual-cased"         
STUDENT_NAME = "distilbert-base-multilingual-cased"   
MAX_LEN = 64

TEACHER_EPOCHS, TEACHER_LR = 5, 4e-5
STUDENT_EPOCHS, STUDENT_LR = 5, 5e-5
BATCH_SIZE = 32
KD_TEMPERATURE, KD_ALPHA = 2.0, 0.5

CUSTOM_OVERSAMPLE = 4

LATENCY_RUNS = 100
LATENCY_WARMUP = 10
