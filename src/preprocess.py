"""Alan B (Klasik NLP): regex normalizasyonu, model kodu çıkarma, tokenization,
minimum edit distance ile ürün adı düzeltme."""
import re
import unicodedata

# Ürün adları sözlüğü (edit distance ile yanlış yazımlar buna düzeltilir)
PRODUCT_LEXICON = [
    # Türkçe
    "bulaşık", "makine", "makinesi", "makinem", "makinemiz", "makineyi", "çamaşır", "kurutma",
    "buzdolabı", "buzdolabım", "buzdolabının", "dondurucu", "fırın", "fırınım", "fırını",
    "mikrodalga", "ocak", "davlumbaz", "klima", "klimam", "kombi", "kombim", "süpürge",
    "süpürgem", "ütü", "televizyon", "ankastre",
    # English
    "dishwasher", "washing", "washer", "machine", "dryer", "fridge", "refrigerator", "freezer",
    "oven", "microwave", "stove", "cooktop", "hood", "conditioner", "boiler", "vacuum",
    "cleaner", "iron", "television", "appliance",
]
LEXICON_SET = set(PRODUCT_LEXICON)

# Model kodu: 2-4 harf + isteğe bağlı tire + 3-6 rakam  (DW-6534, wm8000, RF-12345)
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{2,4})-?(\d{3,6})\b")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # "çokkkk" -> "çokk"
PUNCT_RE = re.compile(r"[^\w\s]|_")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[^\W_]+")
MODEL_TOKEN = "modelcode"


def extract_model_codes(text):
    codes = [f"{a.upper()}-{b}" for a, b in MODEL_CODE_RE.findall(text)]
    text = MODEL_CODE_RE.sub(f" {MODEL_TOKEN} ", text)
    return text, codes


def normalize(text):
    """Regex tabanlı normalizasyon. (normalize edilmiş metin, model kodları) döner."""
    text = unicodedata.normalize("NFC", str(text))
    text = URL_RE.sub(" ", text)
    text, codes = extract_model_codes(text)
    text = text.replace("İ", "i").lower()      # Türkçe büyük İ için
    text = REPEAT_RE.sub(r"\1\1", text)
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text, codes


def tokenize(text):
    return TOKEN_RE.findall(text)


def min_edit_distance(a, b):
    """Levenshtein minimum edit distance (ekleme=silme=değiştirme=1), dinamik programlama."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # silme
                         cur[j - 1] + 1,     # ekleme
                         prev[j - 1] + sub)  # değiştirme
        prev = cur
    return prev[m]


class Preprocessor:
    """normalize -> model kodu çıkar -> tokenize -> ürün adı yazım düzeltme."""

    def __init__(self, known_vocab=None):
        self.known = set(known_vocab or [])

    def correct_token(self, tok):
        if (tok in LEXICON_SET or tok in self.known or tok == MODEL_TOKEN
                or len(tok) < 5 or tok.isdigit()):
            return tok
        max_d = 1 if len(tok) <= 6 else 2
        best, best_d = None, max_d + 1
        for w in PRODUCT_LEXICON:
            if abs(len(w) - len(tok)) > max_d:
                continue
            d = min_edit_distance(tok, w)
            if d < best_d:
                best, best_d = w, d
        return best if best is not None else tok

    def process(self, text):
        norm, codes = normalize(text)
        tokens = tokenize(norm)
        out, corrections = [], []
        for t in tokens:
            c = self.correct_token(t)
            if c != t:
                corrections.append((t, c))
            out.append(c)
        return {
            "normalized": norm,
            "clean": " ".join(out),
            "model_codes": codes,
            "corrections": corrections,
            "has_domain_signal": bool(codes) or any(t in LEXICON_SET for t in out),
        }
