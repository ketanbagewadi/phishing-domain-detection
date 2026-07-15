
# Phishing Domain Detection

A machine learning system to detect phishing URLs/domains. It looks at a URL, pulls out useful features (length, hyphens, subdomains, TLD, etc.), and predicts whether it is **Phishing** or **Not Phishing**.

Accuracy: **~95–97%** (Decision Tree model).

---

## What is a Phishing URL/Domain?

A **phishing URL/domain** is a fake web address created by attackers to trick people into thinking they are visiting a real, trusted website (like a bank, email provider, or online store). The goal is to steal sensitive information — passwords, OTPs, credit card numbers, personal details — or to spread malware.

**Why attackers create them (purpose):**
- Steal login credentials and banking/payment details
- Spread malware or ransomware through fake download links
- Impersonate well-known brands to gain the victim's trust
- Run scams (fake prizes, fake job offers, fake government portals, etc.)
- Harvest personal data for identity theft or resale

**How/why these domains are created (common causes):**
- Cheap, easy domain registration on suspicious or low-cost TLDs (e.g. `.tk`, `.xyz`, `.top`)
- Typosquatting — using domains that look almost identical to a real brand (e.g. `paypa1.com` instead of `paypal.com`)
- Adding extra words/hyphens to look "official" (e.g. `secure-login-paypal-verify.com`)
- Hosting on free/anonymous hosting services that don't verify identity
- Automated bulk domain generation by bots/phishing kits, making thousands of throwaway domains
- Exploiting urgency and fear in social engineering (fake "your account is locked" messages) to get victims to click

Because these domains are created in huge numbers and change constantly, manual detection doesn't scale — which is why this project uses **machine learning** to automatically flag suspicious URLs based on patterns in their structure (length, hyphens, subdomains, TLD, keywords, etc.).

---

## Project Files

| File | What it does |
|---|---|
| `train.py` | Trains the ML models and saves the final model to the `model/` folder |
| `phishing-predictor.py` | Loads the saved model and predicts phishing URLs from a CSV file |
| `alexa.pkl` | List of trusted/known-good TLDs and domains (used to whitelist safe sites) |
| `brand_tokens.pkl` | List of known brand names (used to catch brand impersonation and reduce false positives) |
| `requirements.txt` | Python packages needed to run the project |

> Note: the `model/` folder is not in the repo — it gets created automatically when you run `train.py`.

---

## How It Works

1. **Feature extraction** – Every URL is broken down into simple features: length, number of dots/hyphens/underscores, number of subdomains, TLD, whether the host is an IP address, tokens from the domain and path, etc.
2. **Model training** – These features are fed into a few ML models (Decision Tree, SVC, Logistic Regression, Naive Bayes). The **Decision Tree model is recommended** since it gives the best, most stable accuracy (~95–97%).
3. **Saving the model** – The trained Decision Tree model is saved as a `.pkl` file inside the `model/` folder.
4. **Prediction** – `phishing-predictor.py` loads the saved model along with `alexa.pkl` (trusted domains) and `brand_tokens.pkl` (brand names), then applies a few whitelist checks first, then the ML model, then some extra heuristic rules to give the final label.

---

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- The scripts also use `nltk`'s tokenizer utilities — if you hit an nltk data error, run:

```bash
python -m nltk.downloader punkt
```

---

## Step 1: Prepare the Training Data

`train.py` expects a CSV file named **`all_urls.csv`** in the project root, with two columns:

| Column | Meaning |
|---|---|
| `url` | The full URL or domain (e.g. `example.com/login`) |
| `label` | `0` for a good/legit URL, `1` for a bad/phishing URL |

Example:

```csv
url,label
google.com,0
paypal-secure-login.tk,1
```

The more rows (and the more balanced good vs bad), the better the model performs.

---

## Step 2: Train the Model

Run:

```bash
python train.py
```

This will:
- Load and clean `all_urls.csv`
- Extract features from every URL
- Train Decision Tree, SVC, Logistic Regression, and Naive Bayes models
- Print accuracy/classification reports and confusion matrices for each model
- Save the trained models into the `model/` folder, e.g.:
  - `model/whole_dataset_dt_clf_model.pkl` (Decision Tree — **recommended**)
  - `model/whole_dataset_nb_clf_model.pkl` (Naive Bayes)

The Decision Tree model (`whole_dataset_dt_clf_model.pkl`) is the one used by the predictor script.

---

## Step 3: Run Predictions

Once the model is saved in `model/`, use `phishing-predictor.py` to check a CSV full of domains/URLs:

```bash
python phishing-predictor.py input.csv
```

- `input.csv` should have a column named `query`, `domain`, or `url` (auto-detected).
- If your column has a different name, specify it:

```bash
python phishing-predictor.py input.csv -c my_column_name
```

- Output is saved automatically as `input_results.csv` (or set a custom path):

```bash
python phishing-predictor.py input.csv -o results.csv
```

The output CSV has three columns: `query`, `phishing_flag` (0/1), `phishing_label` (Phishing / Not Phishing).

---

## Quick Start (Summary)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your training data as all_urls.csv (columns: url, label)

# 3. Train the model (saves to model/ folder)
python train.py

# 4. Predict on new URLs
python phishing-predictor.py input.csv
```

---

## Notes

- The Decision Tree model is recommended because it gives the best accuracy (~95–97%) among the models tested here.
- `alexa.pkl` and `brand_tokens.pkl` are large pre-built files used to whitelist trusted domains and detect brand impersonation — they help cut down false positives before the ML model even runs.
- `phishing-predictor.py` de-duplicates queries before predicting, so repeated URLs in the input file are only checked once (faster on large files).
