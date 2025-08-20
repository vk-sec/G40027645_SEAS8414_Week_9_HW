
# SEAS 8414 – Week 9 Homework  
## TESTING.md — Manual Verification + Automation

This guide shows you how to **manually verify** the pipeline on one **DGA-like** and one **legit** domain, and how to **automate** bulk testing with `scripts/test_all.sh`.

---

## 0) Prerequisites

- Python **3.10+** (use your project venv)
- Java **17** on PATH (`java -version`)
- Dependencies installed:
```bash
  pip install -r requirements.txt
````

* A trained model is **not** required; the test steps will train if needed.

* **GenAI note:** If you set `GOOGLE_API_KEY`, the analyzer will call Gemini and print a **prescriptive playbook**. 
* If not set, add `--skip_genai` to analyzer commands (the automation script handles this automatically).

---

## 1) One-off Manual Test (DGA-like domain)

### 1.1 Train (creates `data/` + `model/`)

```bash
python 1_train_and_export.py --rows 6000 --runtime 30
```

Expected:

* `data/dga_dataset_train.csv`
* `model/DGA_Leader.zip`
* `model/leaderboard.csv`

### 1.2 Analyze a DGA-like domain

```bash
# If you have a key:
# export GOOGLE_API_KEY="YOUR_KEY"
python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
# Or without GenAI:
# python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info --skip_genai
```

**What you should see:**

* A **feature vector** (length, entropy)
* A **prediction table** with columns like `predict`, `dga`, `legit`
* A line like:
  `Predicted class: dga  |  P(dga)=0.98xx`
* A **contributions/SHAP** section (length/entropy influence)
* If key is set: a **Gemini playbook** with actionable steps

---

## 2) One-off Manual Test (Legit domain)

### 2.1 Analyze a legit domain

```bash
python 2_analyze_domain.py --domain cnn.com --skip_genai
```

**What you should see:**

* A **feature vector**
* A **prediction table**
* A line like:
  `Predicted class: legit  |  P(dga)=0.0xxx`
* Usually **no playbook** (we only generate one when predicted `dga`)

---

## 3) Automating Tests with `scripts/test_all.sh`

### 3.1 Create the script

Create the folder (if it doesn’t exist) and the script:

```bash
mkdir -p scripts
cat > scripts/test_all.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
PY=python
TRAIN=1_train_and_export.py
ANALYZE=2_analyze_domain.py
MOJO=model/DGA_Leader.zip
ROWS=${ROWS:-6000}
RUNTIME=${RUNTIME:-30}

# Test domain lists (edit here)
DGA_DOMAINS=("kq3v9z7j1x5f8g2h.info" "bad-domain123xyz.biz" "z9q8w7e6r5t4y3u2.top")
LEGIT_DOMAINS=("cnn.com" "mako-shop.com" "example.org")

# If GOOGLE_API_KEY is not set, skip GenAI to avoid errors
GENAI_FLAG="--skip_genai"
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then GENAI_FLAG=""; fi

# ---------- Helpers ----------
train_if_missing() {
  if [[ ! -f "$MOJO" ]]; then
    echo "MOJO not found -> training model..."
    "$PY" "$TRAIN" --rows "$ROWS" --runtime "$RUNTIME"
  else
    echo "MOJO found: $MOJO"
  fi
}

analyze_set() {
  local label="$1"; shift
  local arr=("$@")
  echo ""
  echo "== Running $label set =="
  for d in "${arr[@]}"; do
    echo ""
    echo "---- Analyzing $d ----"
    "$PY" "$ANALYZE" --domain "$d" $GENAI_FLAG
  done
}

# ---------- Main ----------
train_if_missing
analyze_set "DGA"   "${DGA_DOMAINS[@]}"
analyze_set "Legit" "${LEGIT_DOMAINS[@]}"

echo ""
echo "✅ All tests completed."
BASH

chmod +x scripts/test_all.sh
```

### 3.2 How it works

* **Trains automatically** if `model/DGA_Leader.zip` is missing.
* **Runs the analyzer** against two sets:

  * `DGA_DOMAINS` (expected: `dga`)
  * `LEGIT_DOMAINS` (expected: `legit`)
* **GenAI behavior**:

  * If `GOOGLE_API_KEY` is set in your environment, the script includes playbook generation.
  * If not set, it adds `--skip_genai` to each analysis.

### 3.3 Run it

> Use **Git Bash**, **WSL**, or macOS/Linux terminal.

```bash
# (optional) enable GenAI playbooks
# export GOOGLE_API_KEY="YOUR_KEY"

# run the full battery
bash scripts/test_all.sh
```

**Expected output:**

* “MOJO not found → training model…” (first run only), then
* “== Running DGA set ==” followed by each domain result
* “== Running Legit set ==” followed by each domain result
* “✅ All tests completed.”

### 3.4 Customizing

* Add/remove domains by editing the arrays at the top of `test_all.sh`.
* Override dataset size or runtime (e.g., faster iteration):

  ```bash
  ROWS=2000 RUNTIME=20 bash scripts/test_all.sh
  ```

---

## 4) Troubleshooting

* **Java error / cannot start H2O**
  Confirm `java -version` prints Java 17 and it’s on PATH.
* **Import errors**
  Activate the right venv and reinstall:

  ```bash
  pip install -r requirements.txt
  ```
* **GenAI HTTP errors**
  Ensure `GOOGLE_API_KEY` is valid or just run with `--skip_genai`.

---

## 5) Quick Reference

* Train:

  ```bash
  python 1_train_and_export.py --rows 6000 --runtime 30
  ```
* Analyze (DGA-like):

  ```bash
  python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info
  # or: python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info --skip_genai
  ```
* Analyze ( legit ):

  ```bash
  python 2_analyze_domain.py --domain cnn.com --skip_genai
  ```
* Automate:

  ```bash
  bash scripts/test_all.sh
  ```


