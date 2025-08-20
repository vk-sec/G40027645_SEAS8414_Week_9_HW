#!/usr/bin/env python3
"""
Score a single domain with the exported MOJO, print SHAP-style contributions,
and (optionally) generate a prescriptive playbook via Google GenAI.

Usage:
  python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
  python 2_analyze_domain.py --domain foo.com --skip_genai
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h2o

from genai_prescriptions import generate_playbook


# -----------------------------
# Feature helpers
# -----------------------------

def split_sld(domain: str) -> str:
    parts = domain.lower().strip().split(".")
    if len(parts) <= 1:
        return parts[0]
    return parts[-2]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    vals, counts = np.unique(list(s), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze a domain with MOJO and generate XAI+playbook")
    ap.add_argument("--domain", required=True, help="e.g., kq3v9z7j1x5f8g2h.info")
    ap.add_argument("--mojo_path", default="model/DGA_Leader.zip", help="Path to exported MOJO zip")
    ap.add_argument("--google_api_key", default=None, help="Override GOOGLE_API_KEY")
    ap.add_argument("--skip_genai", action="store_true", help="Skip GenAI playbook call")
    args = ap.parse_args()

    domain = args.domain
    sld = split_sld(domain)
    features = {"length": len(sld), "entropy": shannon_entropy(sld)}

    print("\n=== Feature Vector ===")
    for k, v in features.items():
        print(f"{k}: {v}")

    # STEP 1: H2O init
    print("\nSTEP 1: h2o.init()")
    h2o.init(
        ip="127.0.0.1",
        port=54321,
        nthreads=-1,
        max_mem_size="2G",
        strict_version_check=False,
        bind_to_localhost=True,
    )

    # STEP 2: Load MOJO
    print("STEP 2: load MOJO")
    mojo_path = Path(args.mojo_path)
    if not mojo_path.exists():
        raise SystemExit(f"[ERR] MOJO not found at {mojo_path}. Run 1_train_and_export.py first.")
    model = h2o.import_mojo(str(mojo_path.resolve()))

    # STEP 3: Predict
    print("STEP 3: predict")
    hf = h2o.H2OFrame(pd.DataFrame([features]))
    pred = model.predict(hf).as_data_frame()
    print("\n=== Prediction Raw ===")
    print(pred)

    # Try to derive P(dga)
    cols = [c.lower() for c in pred.columns]
    cmap = {c.lower(): c for c in pred.columns}
    p_dga = None

    if "dga" in cmap:
        p_dga = float(pred.loc[0, cmap["dga"]])
    elif "p1" in cmap:
        p_dga = float(pred.loc[0, cmap["p1"]])
    else:
        # assume last prob-like col if present
        prob_cols = [c for c in pred.columns if c.lower().startswith("p")]
        if len(prob_cols) >= 2:
            p_dga = float(pred.loc[0, prob_cols[-1]])
        else:
            p_dga = 1.0 if str(pred.loc[0, "predict"]).lower() in ("1", "dga") else 0.0

    label = "dga" if p_dga >= 0.5 else "legit"
    print(f"\nPredicted class: {label}  |  P(dga)={p_dga:.4f}")

    if label != "dga":
        print("\nNo playbook generated (domain predicted legit).")
        return

    # STEP 4: SHAP-style contributions
    print("\nSTEP 4: SHAP / contributions")
    try:
        contrib = model.predict_contributions(hf).as_data_frame()
        shap_len = float(contrib.loc[0, "length"]) if "length" in contrib.columns else 0.0
        shap_ent = float(contrib.loc[0, "entropy"]) if "entropy" in contrib.columns else 0.0
        bias = float(contrib.loc[0, "BiasTerm"]) if "BiasTerm" in contrib.columns else 0.0

        def push_dir(val: float) -> str:
            return "↑ toward DGA" if val >= 0 else "↓ toward legit"

        xai_findings = (
            f"- Alert: Potential DGA domain detected.\n"
            f"- Domain: '{domain}'\n"
            f"- AI Model Explanation (local SHAP / contributions):\n"
            f"  - Confidence P(dga) = {p_dga*100:.1f}%\n"
            f"  - length = {features['length']} → contribution {shap_len:+.4f} ({push_dir(shap_len)})\n"
            f"  - entropy = {features['entropy']:.3f} → contribution {shap_ent:+.4f} ({push_dir(shap_ent)})\n"
            f"  - bias/intercept = {bias:+.4f}\n"
        )
    except Exception as e:
        xai_findings = (
            f"- Alert: Potential DGA domain detected.\n"
            f"- Domain: '{domain}'\n"
            f"- NOTE: SHAP contributions unavailable ({e}). Proceeding with probability only.\n"
            f"- Confidence P(dga) = {p_dga*100:.1f}%\n"
        )

    print("\n=== XAI Findings (for GenAI) ===")
    print(xai_findings)

    # STEP 5: GenAI bridge (optional)
    if args.skip_genai:
        print("\n[skip_genai] Skipping playbook generation.")
        return

    print("\nSTEP 5: GenAI playbook")
    playbook = generate_playbook(xai_findings, api_key=args.google_api_key)

    print("\n=== Prescriptive Playbook (Gemini) ===")
    print(playbook)


if __name__ == "__main__":
    main()
