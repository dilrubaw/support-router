import argparse
import subprocess
import sys
import time

STEPS = ["prepare_data", "train_baseline", "train_word2vec", "train_ffnn",
         "train_teacher", "distill", "quantize", "evaluate"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
    args = ap.parse_args()
    for step in STEPS[STEPS.index(args.start):]:
        print(f"\n{'=' * 60}\n>>> {step}\n{'=' * 60}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-m", f"src.{step}"], check=True)
        print(f"<<< {step} bitti ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    print("\nHepsi bitti. Sonuçlar: results/results_table.md")
