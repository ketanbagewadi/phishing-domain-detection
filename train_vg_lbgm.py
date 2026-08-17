import pandas as pd
import numpy as np
import pickle
import gc
import os
import io
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from nltk.tokenize import RegexpTokenizer
from urllib.parse import urlparse
import re
import tldextract
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb


# ----------------------------------------------------------------------
# Config knobs
# ----------------------------------------------------------------------
AUDIT_ONLY = False
AUDIT_SAMPLE_SIZE = 2000
SCALE_POS_WEIGHT_MULTIPLIER = 1.0
CV_FOLDS = 3
OUTPUT_REPORT = 'verify_output.txt'

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# ----------------------------------------------------------------------
# Feature helpers (MUST match predictor.py at inference)
# ----------------------------------------------------------------------

def clean_url(url):
    if pd.isna(url) or not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    u = u.replace('[.]', '.').replace('(.)', '.').replace('[dot]', '.')
    u = u.replace('hxxp://', 'http://').replace('hxxps://', 'https://')
    u = u.replace('[', '').replace(']', '')
    printable_count = sum(1 for c in u if c.isprintable() and ord(c) < 128)
    if len(u) == 0 or (printable_count / len(u)) < 0.85:
        return None
    return u


def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        if pd.isna(url) or not isinstance(url, str):
            return {"netloc": None, "path": ""}
        url = str(url).strip()
        if not url:
            return {"netloc": None, "path": ""}
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        parsed_url = urlparse(f"http://{url}" if no_scheme else url)
        return {"netloc": parsed_url.netloc, "path": parsed_url.path}
    except (ValueError, TypeError, AttributeError):
        return {"netloc": None, "path": ""}


def get_num_subdomains(netloc: str) -> int:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return 0
        subdomain = tldextract.extract(netloc).subdomain
        return subdomain.count('.') + 1 if subdomain else 0
    except Exception:
        return 0


def get_registered_domain(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        return tldextract.extract(netloc).domain.lower()
    except Exception:
        return ""


def get_full_domain(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        ext = tldextract.extract(netloc)
        if not ext.domain:
            return ""
        return f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()
    except Exception:
        return ""


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


def tokenize_domain(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        ext = tldextract.extract(netloc)
        return " ".join(tokenizer.tokenize(f"{ext.subdomain}.{ext.domain}"))
    except Exception:
        return ""


def tokenize_path(path: str) -> str:
    try:
        if pd.isna(path) or not isinstance(path, str):
            return ""
        return " ".join(tokenizer.tokenize(path))
    except Exception:
        return ""


class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()


def load_domain_set_from_pickle(path, preferred_cols):
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, pd.DataFrame):
        col = None
        for c in preferred_cols:
            if c in obj.columns:
                col = c
                break
        if col is None:
            col = obj.columns[0]
        values = obj[col].dropna().astype(str).str.strip().str.lower()
        return set(values)
    elif isinstance(obj, (set, list, tuple)):
        return set(str(v).strip().lower() for v in obj)
    else:
        raise TypeError(f"Unrecognized pickle format in {path}: {type(obj)}")


def get_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=['good', 'bad'],
                                    output_dict=True, zero_division=0)
    bad = report.get('bad', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
    return {
        'accuracy': report.get('accuracy', 0.0),
        'precision_bad': bad['precision'],
        'recall_bad': bad['recall'],
        'f1_bad': bad['f1-score']
    }


def diagnose_fit(train_metrics, test_metrics, cv_mean=None, cv_std=None):
    train_f1 = train_metrics['f1_bad']
    test_f1 = test_metrics['f1_bad']
    gap = train_f1 - test_f1
    log(f"  Train F1 (bad): {train_f1:.3f}  |  Test F1 (bad): {test_f1:.3f}  |  Gap: {gap:.3f}")
    if cv_mean is not None:
        log(f"  {CV_FOLDS}-fold CV F1 (bad): {cv_mean:.3f} (+/- {cv_std:.3f})")
    if train_f1 < 0.55 and test_f1 < 0.55:
        verdict = "UNDERFITTING - model too weak, needs more capacity."
    elif gap > 0.12:
        verdict = "OVERFITTING - train/test gap too large, needs more regularization."
    elif gap <= 0.08 and test_f1 >= 0.65:
        verdict = "GENERALIZED - low variance, low bias. Good fit for production."
    else:
        verdict = "ACCEPTABLE - moderate gap/scores, usable but could be tuned further."
    log(f"  Verdict: {verdict}")
    return verdict


def manual_cv_f1(pipeline_builder, X, y, n_folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipe = pipeline_builder()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = pipe.predict(X.iloc[val_idx])
        rep = classification_report(y.iloc[val_idx], preds, labels=[0, 1], target_names=['good', 'bad'],
                                     output_dict=True, zero_division=0)
        f1 = rep.get('bad', {}).get('f1-score', 0.0)
        scores.append(f1)
        log(f"    fold {fold_i}/{n_folds}: F1(bad)={f1:.3f}")
        del pipe
        gc.collect()
    return np.array(scores)


if __name__ == '__main__':
    df = pd.read_csv('all_urls.csv')
    log(f"Loaded {len(df)} rows")
    df = df.dropna(subset=['url'])
    df['url'] = df['url'].astype(str)
    df['label'] = df['label'].map({0: 'good', 1: 'bad'})

    before = len(df)
    df['url'] = df['url'].progress_apply(clean_url)
    df = df.dropna(subset=['url'])
    log(f"Dropped {before - len(df)} garbage rows")

    df_grp = df.groupby("url")["label"].agg(list).reset_index()
    del df
    gc.collect()
    df_grp["parsed"] = df_grp["url"].progress_apply(parse_url)
    df_grp["label"] = df_grp["label"].apply(lambda x: 'bad' if 'bad' in x else 'good')
    df_grp = pd.concat([df_grp.drop(['parsed'], axis=1), df_grp['parsed'].apply(pd.Series)], axis=1)
    df_grp = df_grp.dropna(subset=['netloc'])
    df_grp = df_grp[df_grp['netloc'] != '']
    log(f"After cleaning netloc: {len(df_grp)} rows")

    df_grp["tld"] = df_grp.netloc.progress_apply(lambda nl: tldextract.extract(nl).suffix or 'None')
    df_grp["length"] = df_grp.url.str.len()
    df_grp["is_ip"] = df_grp.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
    df_grp['domain_hyphens'] = df_grp.netloc.str.count('-')
    df_grp['domain_underscores'] = df_grp.netloc.str.count('_')
    df_grp['path_hyphens'] = df_grp.path.fillna('').str.count('-')
    df_grp['path_underscores'] = df_grp.path.fillna('').str.count('_')
    df_grp['slashes'] = df_grp.path.fillna('').str.count('/')
    df_grp['full_stops'] = df_grp.path.fillna('').str.count('.')
    df_grp['num_subdomains'] = df_grp['netloc'].progress_apply(get_num_subdomains)
    df_grp['domain'] = df_grp['netloc'].progress_apply(get_registered_domain)
    df_grp['full_domain'] = df_grp['netloc'].progress_apply(get_full_domain)
    df_grp['domain_tokens'] = df_grp['netloc'].progress_apply(tokenize_domain)
    df_grp['path_tokens'] = df_grp['path'].fillna('').progress_apply(tokenize_path)

    log(f"\nOriginal label distribution:\n{df_grp['label'].value_counts().to_string()}")

    brand_tokens = set()
    try:
        brand_tokens = load_domain_set_from_pickle('brand_tokens.pkl', ['brand', 'main_domain', 'domain'])
        log(f"Loaded {len(brand_tokens)} brand tokens (main domain, no TLD) from brand_tokens.pkl")
    except FileNotFoundError:
        log("brand_tokens.pkl not found, will build from good-labeled data only")

    alexa_full_domains = set()
    try:
        alexa_full_domains = load_domain_set_from_pickle('alexa.pkl', ['domain'])
        log(f"Loaded {len(alexa_full_domains)} full domains from alexa.pkl")
    except FileNotFoundError:
        log("alexa.pkl not found, skipping alexa whitelist")

    good_domains = df_grp[df_grp['label'] == 'good']['domain'].value_counts()
    derived_brand_tokens = set(good_domains[(good_domains.index.str.len() >= 3) &
                                             (good_domains.index != '')].index)
    brand_tokens = brand_tokens | derived_brand_tokens

    with open('brand_tokens.pkl', 'wb') as f:
        pickle.dump(brand_tokens, f)
    log(f"Saved {len(brand_tokens)} total brand tokens (as a set, for predictor.py)")

    brand_mask = df_grp['domain'].isin(brand_tokens) & (df_grp['label'] != 'bad')
    alexa_mask = df_grp['full_domain'].isin(alexa_full_domains) & (df_grp['label'] != 'bad')
    combined_mask = brand_mask | alexa_mask
    log(f"Brand/alexa-relabeled rows: {combined_mask.sum()} | "
        f"Confirmed 'bad' rows preserved: {(df_grp['label'] == 'bad').sum()}")

    audit_cols = ['url', 'netloc', 'domain', 'full_domain']
    brand_sample = df_grp.loc[brand_mask, audit_cols].sample(
        n=min(AUDIT_SAMPLE_SIZE, brand_mask.sum()), random_state=42
    ).assign(matched_via='brand_tokens') if brand_mask.sum() else pd.DataFrame(columns=audit_cols + ['matched_via'])
    alexa_sample = df_grp.loc[alexa_mask, audit_cols].sample(
        n=min(AUDIT_SAMPLE_SIZE, alexa_mask.sum()), random_state=42
    ).assign(matched_via='alexa') if alexa_mask.sum() else pd.DataFrame(columns=audit_cols + ['matched_via'])
    audit_df = pd.concat([brand_sample, alexa_sample], ignore_index=True)
    audit_df.to_csv('relabel_audit_sample.csv', index=False)
    log(f"Saved {len(audit_df)} rows to relabel_audit_sample.csv for manual review")

    if AUDIT_ONLY:
        log("AUDIT_ONLY=True -> stopping before training.")
        with open(OUTPUT_REPORT, 'w') as f:
            f.write("\n".join(report_lines))
        raise SystemExit(0)

    df_grp.loc[combined_mask, 'label'] = 'good'
    log(f"\nFinal label distribution after relabeling:\n{df_grp['label'].value_counts().to_string()}")

    y = df_grp['label']
    X = df_grp[['length', 'tld', 'is_ip', 'domain_hyphens', 'domain_underscores',
                'path_hyphens', 'path_underscores', 'slashes', 'full_stops',
                'num_subdomains', 'domain_tokens', 'path_tokens']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    del df_grp
    gc.collect()

    LABEL_MAP = {'good': 0, 'bad': 1}
    y_train = y_train.map(LABEL_MAP)
    y_test = y_test.map(LABEL_MAP)

    numeric_features = ['length', 'domain_hyphens', 'domain_underscores', 'path_hyphens',
                         'path_underscores', 'slashes', 'full_stops', 'num_subdomains']
    categorical_features = ['tld', 'is_ip']

    def make_preprocessor():
        return ColumnTransformer(transformers=[
            ('num', Pipeline([('scaler', MinMaxScaler())]), numeric_features),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
            ('domvec', Pipeline([('con', Converter()), ('tf', TfidfVectorizer(max_features=2000))]), ['domain_tokens']),
            ('pathvec', Pipeline([('con', Converter()), ('tf', TfidfVectorizer(max_features=2000))]), ['path_tokens']),
        ])

    n_good = (y_train == 0).sum()
    n_bad = (y_train == 1).sum()
    scale_pos_weight = (n_good / n_bad) * SCALE_POS_WEIGHT_MULTIPLIER
    log(f"\nn_good={n_good}  n_bad={n_bad}  imbalance_ratio={n_good/n_bad:.2f}:1  "
        f"scale_pos_weight={scale_pos_weight:.2f} (multiplier={SCALE_POS_WEIGHT_MULTIPLIER})")

    def build_xgb():
        return Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', xgb.XGBClassifier(
                tree_method='hist',
                device='cuda',
                eval_metric='logloss',
                n_estimators=200,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            ))
        ])

    def build_lgbm():
        def _lgbm_classifier(device):
            return lgb.LGBMClassifier(
                device=device,
                gpu_use_dp=True if device == 'cuda' else False,
                objective='binary',
                n_estimators=200,
                num_leaves=31,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=30,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
                random_state=42
            )

        # Probe: plain PyPI lightgbm wheels are NOT built with CUDA support.
        # Test on a tiny dummy fit first instead of discovering this after
        # a long real .fit() call.
        use_cuda = True
        try:
            probe = lgb.LGBMClassifier(device='cuda', n_estimators=1, verbosity=-1)
            probe.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 1, 0, 1]))
        except Exception as e:
            use_cuda = False
            log(f"WARNING: LightGBM CUDA build not available ({e}). "
                f"Falling back to CPU for LightGBM. To fix properly: "
                f"pip uninstall lightgbm -y --break-system-packages && "
                f"pip install lightgbm --break-system-packages "
                f"--config-settings=cmake.define.USE_CUDA=ON --no-binary lightgbm")

        device = 'cuda' if use_cuda else 'cpu'
        return Pipeline(steps=[
            ('preprocessor', make_preprocessor()),
            ('classifier', _lgbm_classifier(device))
        ])

    model_builders = {"XGBoost": build_xgb, "LightGBM": build_lgbm}
    trained_models = {}
    summary = {}

    for name, builder in model_builders.items():
        log(f"\n{'='*70}\nTraining {name}\n{'='*70}")
        model = builder()
        model.fit(X_train, y_train)
        trained_models[name] = model
        gc.collect()

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        log(f"\n--- {name}: Test set classification report ---")
        log(classification_report(y_test, test_preds, labels=[0, 1], target_names=['good', 'bad'], zero_division=0))
        log(f"--- {name}: Confusion matrix (rows=true, cols=pred, order=[good,bad]) ---")
        log(str(confusion_matrix(y_test, test_preds, labels=[0, 1])))

        log(f"\n--- {name}: Train set classification report (overfit/underfit check) ---")
        log(classification_report(y_train, train_preds, labels=[0, 1], target_names=['good', 'bad'], zero_division=0))

        train_metrics = get_metrics(y_train, train_preds)
        test_metrics = get_metrics(y_test, test_preds)

        log(f"\n--- {name}: {CV_FOLDS}-fold cross-validation (fresh model per fold) ---")
        cv_scores = manual_cv_f1(builder, X_train, y_train, n_folds=CV_FOLDS)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        log(f"\n--- {name}: Overfitting / Underfitting diagnosis ---")
        verdict = diagnose_fit(train_metrics, test_metrics, cv_mean, cv_std)

        classes = model.named_steps['classifier'].classes_
        bad_idx = list(classes).index(1)
        probs_bad = model.predict_proba(X_test)[:, bad_idx]

        log(f"\n--- {name}: Threshold sweep (precision/recall/f1 for 'bad') ---")
        threshold_rows = []
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            preds = np.where(probs_bad >= t, 1, 0)
            rep = classification_report(y_test, preds, labels=[0, 1], target_names=['good', 'bad'],
                                         output_dict=True, zero_division=0)
            bad = rep.get('bad', {})
            line = (f"  t={t}: precision={bad.get('precision',0):.3f} "
                    f"recall={bad.get('recall',0):.3f} f1={bad.get('f1-score',0):.3f} "
                    f"accuracy={rep.get('accuracy',0):.3f}")
            log(line)
            threshold_rows.append((t, bad.get('precision', 0), bad.get('recall', 0), bad.get('f1-score', 0)))

        summary[name] = {
            'train': train_metrics, 'test': test_metrics,
            'cv_mean': cv_mean, 'cv_std': cv_std, 'verdict': verdict,
            'thresholds': threshold_rows
        }
        gc.collect()

    log(f"\n{'='*90}")
    log("FINAL SUMMARY - Accuracy / Precision / Recall / F1 (bad class) + generalization verdict")
    log(f"{'='*90}")
    log(f"{'Model':<12}{'Accuracy':<10}{'Train F1':<10}{'Test F1':<10}{'Prec':<8}{'Rec':<8}{'CV F1':<10}{'Verdict'}")
    for name, m in summary.items():
        log(f"{name:<12}{m['test']['accuracy']:<10.3f}{m['train']['f1_bad']:<10.3f}"
            f"{m['test']['f1_bad']:<10.3f}{m['test']['precision_bad']:<8.3f}"
            f"{m['test']['recall_bad']:<8.3f}{m['cv_mean']:<10.3f}{m['verdict']}")
    log(f"{'='*90}")

    best_model_name = max(summary, key=lambda k: summary[k]['test']['f1_bad'])
    log(f"\nBest model by test F1 (bad class): {best_model_name}")
    log("For production (2-4k domains/min, no false positives), use this model's "
        "predict_proba() with a raised threshold (0.8-0.9) - see the threshold sweep "
        "above for the exact precision/recall tradeoff at each level.")

    os.makedirs('model', exist_ok=True)

    with open('model/whole_dataset_xgb_clf_model.pkl', 'wb') as f:
        pickle.dump(trained_models['XGBoost'], f)
    with open('model/whole_dataset_lgbm_clf_model.pkl', 'wb') as f:
        pickle.dump(trained_models['LightGBM'], f)

    for fname in ['whole_dataset_xgb_clf_model.pkl', 'whole_dataset_lgbm_clf_model.pkl']:
        size_mb = os.path.getsize(f'model/{fname}') / (1024 ** 2)
        log(f"Saved model/{fname} ({size_mb:.2f} MB)")

    with open(OUTPUT_REPORT, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nFull verification report written to {OUTPUT_REPORT}")
