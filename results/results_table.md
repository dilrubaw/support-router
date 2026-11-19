| Model | Hassasiyet | Accuracy (tüm test) | In-scope accuracy | OOS recall | Beyaz eşya acc. (TR+EN) | Türkçe acc. | Eşik (val) | Boyut (MB) | Gecikme ort. (ms) | Gecikme p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1) TF-IDF + LogReg (klasik baseline) | FP64 (sklearn) | 82.0% | 90.3% | 44.6% | 55.0% | 64.0% | 0.05 | 41.6 | 18.2 | 21.1 |
| 2) word2vec + FFNN (SGD) | FP32 | 81.2% | 88.3% | 49.2% | 50.0% | 52.0% | 0.25 | 2.1 | 0.2 | 0.3 |
| 3) mBERT fine-tuned (Transformer, FP32) | FP32 | 86.7% | 94.5% | 51.4% | 80.0% | 84.0% | 0.1 | 711.9 | 109.3 | 135.1 |
| 4a) DistilmBERT öğrenci (distilled, FP32) | FP32 | 88.1% | 93.7% | 62.5% | 70.0% | 76.0% | 0.3 | 541.8 | 56.3 | 71.6 |
| 4b) DistilmBERT öğrenci INT8 (quantized) ★ | INT8 (dynamic quantization) | 86.5% | 92.3% | 59.8% | 62.5% | 72.0% | 0.1 | 412.3 | 20.0 | 26.3 |

CPU: Intel(R) Xeon(R) CPU @ 2.00GHz
CPU thread sayısı (torch): 1
Platform: Linux-6.6.159+-x86_64-with-glibc2.39
Python 3.13.16, torch 2.14.1+cu130
Gecikme: CPU, batch=1, 100 çalıştırma ortalaması (10 ısınma sonrası), tokenization dahil.
Test seti: 5540 mesaj (CLINC150 test 5500 + kendi verimiz 40), oos: 1000
