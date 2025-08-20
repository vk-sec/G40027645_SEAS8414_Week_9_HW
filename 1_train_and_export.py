#!/usr/bin/env python3
"""
Train a DGA detector with H2O AutoML (fallback to GBM), then export a MOJO.
Also writes the synthetic training CSV for grading/auditing.

Usage:
  python 1_train_and_export.py --rows 6000 --runtime 30
"""

import argparse
import os
import random
import string
from pathlib import Path

import h2o
import numpy as np
import pandas as pd

# -----------------------------
# Synthetic data & features
# -----------------------------

ALPHABET = string.ascii_lowercase + string.digits
COMMON_TLDS = ["com", "net", "org", "io", "ai", "co", "info", "biz", "site", "top"]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    vals, counts = np.unique(list(s), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def random_legit_label() -> str:
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    length = random.randint(5, 12)
    name = []
    for i in range(length):
        name.append(random.choice(consonants if i % 2 == 0 else vowels))
    if random.random() < 0.2:
        name.append("-")
        for i in range(random.randint(2, 4)):
            name.append(random.choice(consonants if i % 2 == 0 else vowels))
    return "".join(name)


def random_dga_label() -> str:
    length = random.randint(12, 26)
    s = "".join(random.choice(ALPHABET) for _ in range(length))
    if random.random() < 0.3:
        pos = random.randint(3, max(3, length - 3))
        s = s[:pos] + "-" + s[pos:]
    return s


def make_domain(sld: str) -> str:
    return f"{sld}.{random.choice(COMMON_TLDS)}"


def synth_dataset(n_legit: int, n_dga: int) -> pd.DataFrame:
    rows = []
    for _ in range(n_legit):
        sld = random_legit_label()
        dom = make_domain(sld)
        rows.append(
            {"domain": dom, "length": len(sld), "entropy": shannon_entropy(sld), "label": "legit"}
        )
    for _ in range(n_dga):
        sld = random_dga_label()
        dom = make_domain(sld)
        rows.append(
            {"domain": dom, "length": len(sld), "entropy": shannon_entropy(sld), "label": "dga"}
        )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser(description="Train H2O AutoML DGA model and export MOJO")
    ap.add_argument(
        "--rows", type=int, default=6000, help="rows to synthesize (balanced legit/dga)"
    )
    ap.add_argument("--runtime", type=int, default=30, help="AutoML max_runtime_secs")
    args = ap.parse_args()

    # dirs
    Path("data").mkdir(exist_ok=True)
    Path("model").mkdir(exist_ok=True)

    # data
    n_legit = args.rows // 2
    n_dga = args.rows - n_legit
    df = synth_dataset(n_legit, n_dga)
    csv_path = Path("data") / "dga_dataset_train.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote dataset: {csv_path} (rows={len(df)})")

    # STEP 1: H2O init (single node / localhost)
    print("STEP 1: h2o.init()")
    h2o.init(
        ip="127.0.0.1",
        port=54321,
        nthreads=-1,
        max_mem_size="2G",
        strict_version_check=False,
        bind_to_localhost=True,
    )

    # STEP 2: to H2OFrame
    print("STEP 2: building H2OFrame")
    hf = h2o.H2OFrame(df)
    y = "label"
    x = ["length", "entropy"]
    hf[y] = hf[y].asfactor()

    # STEP 3: AutoML (with safe bounds) or fallback GBM
    print("STEP 3: AutoML training")
    from h2o.automl import H2OAutoML

    try:
        aml = H2OAutoML(
            max_runtime_secs=args.runtime,
            max_models=12,
            seed=42,
            sort_metric="AUC",
            balance_classes=True,
            nfolds=5,
            exclude_algos=[
                "DeepLearning",
                "XGBoost",
                "StackedEnsemble",
            ],  # avoid slow/hanging algos on some setups
        )
        aml.train(x=x, y=y, training_frame=hf)
        leader = aml.leader
        lb_df = aml.leaderboard.as_data_frame()
        print("🏁 AutoML Leader:", leader.model_id)
    except Exception as e:
        print(f"[WARN] AutoML error: {e}\n→ Falling back to a small GBM to guarantee a MOJO.")
        from h2o.estimators import H2OGradientBoostingEstimator

        gbm = H2OGradientBoostingEstimator(
            ntrees=60, max_depth=4, learn_rate=0.1, balance_classes=True, seed=42
        )
        gbm.train(x=x, y=y, training_frame=hf)
        leader = gbm
        lb_df = pd.DataFrame([{"model_id": gbm.model_id, "algo": "GBM (fallback)"}])
        print("🏁 GBM Leader:", leader.model_id)

    # STEP 4: Export
    print("STEP 4: Export MOJO + leaderboard")
    lb_path = Path("model") / "leaderboard.csv"
    lb_df.to_csv(lb_path, index=False)
    print(f"📊 Leaderboard saved to: {lb_path}")

    mojo_tmp = leader.download_mojo(path=str(Path("model").resolve()), get_genmodel_jar=False)
    final_mojo = Path("model") / "DGA_Leader.zip"
    os.replace(mojo_tmp, final_mojo)
    print(f"📦 MOJO exported to: {final_mojo}")

    print("\nAll done. Now run 2_analyze_domain.py to score a domain.")


if __name__ == "__main__":
    main()
