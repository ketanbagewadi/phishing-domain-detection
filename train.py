import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import tldextract
from typing import Dict, List, Optional


# ----------------------------------------------------------------------
# Feature helpers (MUST match what predictor.py computes at inference)
# ----------------------------------------------------------------------

def extract_features(url):
    try:
        if pd.isna(url) or not isinstance(url, str):
            return {'length_url': 0, 'length_hostname': 0, 'ip': 0,
                     'nb_dots': 0, 'nb_hyphens': 0, 'nb_at': 0}

        parsed_url = urlparse(url)
        features = {}
        features['length_url'] = len(url)
        features['length_hostname'] = len(parsed_url.hostname) if parsed_url.hostname else 0
        features['ip'] = 1 if parsed_url.hostname and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", parsed_url.hostname) else 0
        features['nb_dots'] = parsed_url.hostname.count('.') if parsed_url.hostname else 0
        features['nb_hyphens'] = parsed_url.hostname.count('-') if parsed_url.hostname else 0
        features['nb_at'] = url.count('@')
        return features
    except (ValueError, TypeError, AttributeError):
        return {'length_url': len(str(url)) if url else 0, 'length_hostname': 0, 'ip': 0,
                 'nb_dots': 0, 'nb_hyphens': 0, 'nb_at': str(url).count('@') if url else 0}


def clean_url(url):
    """De-defang common threat-intel obfuscation and drop non-URL garbage rows."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    u = url.strip()
    if not u:
        return None
    # de-defang common obfuscation patterns seen in threat feeds
    u = u.replace('[.]', '.').replace('(.)', '.').replace('[dot]', '.')
    u = u.replace('hxxp://', 'http://').replace('hxxps://', 'https://')
    u = u.replace('[', '').replace(']', '')
    # drop rows that are mostly non-printable / binary garbage, not real URLs
    printable_count = sum(1 for c in u if c.isprintable() and ord(c) < 128)
    if len(u) == 0 or (printable_count / len(u)) < 0.85:
        return None
    return u


def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        if pd.isna(url) or not isinstance(url, str):
            return {"scheme": None, "netloc": None, "path": "", "params": "", "query": "", "fragment": ""}

        url = str(url).strip()
        if not url:
            return {"scheme": None, "netloc": None, "path": "", "params": "", "query": "", "fragment": ""}

        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {"scheme": None, "netloc": parsed_url.netloc, "path": parsed_url.path,
                     "params": parsed_url.params, "query": parsed_url.query, "fragment": parsed_url.fragment}
        else:
            parsed_url = urlparse(url)
            return {"scheme": parsed_url.scheme, "netloc": parsed_url.netloc, "path": parsed_url.path,
                     "params": parsed_url.params, "query": parsed_url.query, "fragment": parsed_url.fragment}
    except (ValueError, TypeError, AttributeError):
        # silently skip - malformed/garbage rows are counted and dropped upstream, not logged per-row
        return {"scheme": None, "netloc": None, "path": "", "params": "", "query": "", "fragment": ""}


def combine_labels(labels: List[str]) -> str:
    return ''.join(labels)


def extract_tld(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return 'None'
        return tldextract.extract(netloc).suffix
    except Exception:
        return 'None'


def get_num_subdomains(netloc: str) -> int:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return 0
        subdomain = tldextract.extract(netloc).subdomain
        if subdomain == "":
            return 0
        return subdomain.count('.') + 1
    except Exception:
        return 0


def get_registered_domain(netloc: str) -> str:
    """Main domain without TLD/subdomain, e.g. 'selectorshub' from 'shub.selectorshub.info'."""
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        return tldextract.extract(netloc).domain.lower()
    except Exception:
        return ""


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


def tokenize_domain(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        split_domain = tldextract.extract(netloc)
        no_tld = str(split_domain.subdomain + '.' + split_domain.domain)
        return " ".join(map(str, tokenizer.tokenize(no_tld)))
    except Exception:
        return ""


def tokenize_path(path: str) -> str:
    try:
        if pd.isna(path) or not isinstance(path, str):
            return ""
        return " ".join(map(str, tokenizer.tokenize(path)))
    except Exception:
        return ""


class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


def get_metrics(y_true, y_pred):
    """Return precision/recall/f1 for the 'bad' (phishing) class and overall accuracy."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    bad = report.get('bad', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
    return {
        'accuracy': report.get('accuracy', 0.0),
        'precision_bad': bad['precision'],
        'recall_bad': bad['recall'],
        'f1_bad': bad['f1-score']
    }


def diagnose_fit(train_metrics, test_metrics, cv_mean=None, cv_std=None):
    """Simple rule-of-thumb diagnosis: overfitting vs underfitting vs generalized."""
    train_f1 = train_metrics['f1_bad']
    test_f1 = test_metrics['f1_bad']
    gap = train_f1 - test_f1

    print(f"  Train F1 (bad): {train_f1:.3f}  |  Test F1 (bad): {test_f1:.3f}  |  Gap: {gap:.3f}")
    if cv_mean is not None:
        print(f"  5-fold CV F1 (bad): {cv_mean:.3f} (+/- {cv_std:.3f})")

    if train_f1 < 0.60 and test_f1 < 0.60:
        verdict = "UNDERFITTING - model is too simple / not learning the pattern well. Try more features, less regularization, or a more expressive model."
    elif gap > 0.15:
        verdict = "OVERFITTING - model memorizes training data but doesn't generalize. Reduce model complexity (lower max_depth, higher min_samples_leaf) or add more training data."
    elif gap <= 0.10 and test_f1 >= 0.75:
        verdict = "GENERALIZED - train and test performance are close and both reasonably high. Good fit."
    else:
        verdict = "ACCEPTABLE but keep an eye on it - moderate gap or moderate scores. Consider more tuning."

    print(f"  Verdict: {verdict}")
    return verdict


def results(name: str, model: BaseEstimator, X_train, y_train, X_test, y_test, run_cv=True) -> None:
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print(f"\n=== {name} ===")
    print(f"--- Test set classification report ---")
    print(name + " test accuracy: %.3f" % model.score(X_test, y_test))
    print(classification_report(y_test, test_preds, zero_division=0))

    print(f"--- Train set classification report (for overfit/underfit check) ---")
    print(name + " train accuracy: %.3f" % model.score(X_train, y_train))
    print(classification_report(y_train, train_preds, zero_division=0))

    train_metrics = get_metrics(y_train, train_preds)
    test_metrics = get_metrics(y_test, test_preds)

    cv_mean, cv_std = None, None
    if run_cv:
        def bad_f1_scorer(estimator, X, y):
            preds = estimator.predict(X)
            rep = classification_report(y, preds, output_dict=True, zero_division=0)
            return rep.get('bad', {}).get('f1-score', 0.0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=bad_f1_scorer, n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    print(f"--- Overfitting / Underfitting diagnosis ---")
    diagnose_fit(train_metrics, test_metrics, cv_mean, cv_std)

    labels = ['good', 'bad']
    conf_matrix = confusion_matrix(y_test, test_preds, labels=labels)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Greens')
    plt.title("Confusion Matrix for " + name)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

    return {'train': train_metrics, 'test': test_metrics, 'cv_mean': cv_mean, 'cv_std': cv_std}


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load & clean data
    # ------------------------------------------------------------------
    df = pd.read_csv('all_urls.csv')
    print(f"Loaded dataset with {len(df)} rows")

    df = df.dropna(subset=['url'])
    print(f"After removing missing URLs: {len(df)} rows")

    df['url'] = df['url'].astype(str)
    df['label'] = df['label'].map({0: 'good', 1: 'bad'})
    print("Label distribution:")
    print(df['label'].value_counts())

    # ------------------------------------------------------------------
    # Clean URLs: de-defang obfuscated feed entries ([.] -> ., hxxp -> http)
    # and drop rows that are binary/corrupted garbage, not real URLs.
    # ------------------------------------------------------------------
    before_clean = len(df)
    df['url'] = df['url'].apply(clean_url)
    df = df.dropna(subset=['url'])
    print(f"Dropped {before_clean - len(df)} garbage/unparseable rows during cleaning "
          f"({len(df)} rows remain)")

    # ------------------------------------------------------------------
    # Group / parse
    # ------------------------------------------------------------------
    df_grp = df.groupby("url")["label"].agg(list).reset_index()
    df_grp["parsed_url"] = df_grp["url"].apply(parse_url)
    df_grp["label"] = df_grp["label"].apply(lambda labels: ''.join(map(str, labels)))
    df_grp["label"] = df_grp["label"].apply(lambda x: x[0] if x else '')
    df_grp['label'] = df_grp['label'].str.lower()
    df_grp['label'] = df_grp['label'].replace({'g': 'good', 'b': 'bad'})

    df_grp = pd.concat([
        df_grp.drop(['parsed_url'], axis=1),
        df_grp['parsed_url'].apply(pd.Series)
    ], axis=1)

    df_grp = df_grp.dropna(subset=['netloc'])
    df_grp = df_grp[df_grp['netloc'] != '']
    df_grp = df_grp[df_grp['netloc'].notna()]
    print(f"After removing invalid netloc: {len(df_grp)} rows")

    # ------------------------------------------------------------------
    # Feature engineering (matches predictor.py's inference-time logic)
    # ------------------------------------------------------------------
    df_grp["tld"] = df_grp.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
    df_grp['tld'] = df_grp['tld'].replace('', 'None')
    df_grp["length"] = df_grp.url.str.len()
    df_grp["is_ip"] = df_grp.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")

    df_grp['domain_hyphens'] = df_grp.netloc.str.count('-')
    df_grp['domain_underscores'] = df_grp.netloc.str.count('_')
    df_grp['path_hyphens'] = df_grp.path.fillna('').str.count('-')
    df_grp['path_underscores'] = df_grp.path.fillna('').str.count('_')
    df_grp['slashes'] = df_grp.path.fillna('').str.count('/')
    df_grp['full_stops'] = df_grp.path.fillna('').str.count('.')
    df_grp['num_subdomains'] = df_grp['netloc'].apply(get_num_subdomains)
    df_grp['domain'] = df_grp['netloc'].apply(get_registered_domain)
    df_grp['domain_tokens'] = df_grp['netloc'].apply(tokenize_domain)
    df_grp['path_tokens'] = df_grp['path'].fillna('').apply(tokenize_path)

    # ------------------------------------------------------------------
    # BRAND / MAIN-DOMAIN WHITELIST
    # Rule: if the "main domain" (registered domain, no TLD/subdomain)
    # matches a known legit brand -> treat as NOT phishing.
    # e.g. "shub.selectorshub.info" -> main domain = "selectorshub"
    #      if "selectorshub" is in alexa_main_domains.csv -> skip/legit
    # ------------------------------------------------------------------
    ALEXA_BRANDS_CSV = 'alexa_main_domains.csv'
    alexa_brand_domains = set()
    try:
        alexa_df = pd.read_csv(ALEXA_BRANDS_CSV)
        # Accept whatever the column is named: 'domain', 'main_domain', 'brand', or first column
        candidate_cols = ['main_domain', 'domain', 'brand', 'brand_name']
        col = next((c for c in candidate_cols if c in alexa_df.columns), alexa_df.columns[0])
        alexa_brand_domains = set(
            alexa_df[col].dropna().astype(str).str.strip().str.lower()
        )
        print(f"Loaded {len(alexa_brand_domains)} brand main-domains from {ALEXA_BRANDS_CSV}")
    except FileNotFoundError:
        print(f"WARNING: {ALEXA_BRANDS_CSV} not found. Skipping brand whitelist merge.")

    # Also derive brand tokens from 'good'-labeled data as before (backup source)
    MIN_BRAND_LEN = 3
    brand_domain_counts = df_grp[df_grp['label'] == 'good']['domain'].value_counts()
    derived_brand_tokens = set(
        brand_domain_counts[
            (brand_domain_counts.index.str.len() >= MIN_BRAND_LEN) &
            (brand_domain_counts.index != '')
        ].index
    )

    # Union: alexa_main_domains.csv brands + derived brands from good URLs
    brand_name_tokens = alexa_brand_domains | derived_brand_tokens
    print(f"Total brand tokens (union): {len(brand_name_tokens)}")

    with open('brand_tokens.pkl', 'wb') as f:
        pickle.dump(brand_name_tokens, f)
    print("brand_tokens.pkl saved as a set of main-domain strings")

    # Force-label rows whose main domain is a known brand as 'good' —
    # BUT NEVER override a row that's already confirmed 'bad' (phishing).
    # A brand name appearing as the main domain doesn't guarantee legitimacy
    # (e.g. a phishing site using a common/generic word that's coincidentally
    # in the huge alexa_main_domains.csv list). Only relabel rows that were
    # already 'good' or ambiguous — this preserves your real phishing signal.
    brand_mask = df_grp['domain'].isin(brand_name_tokens) & (df_grp['label'] != 'bad')
    print(f"Rows matching a known brand main-domain (excluding confirmed bad): {brand_mask.sum()}")
    print(f"Confirmed 'bad' rows preserved untouched: {(df_grp['label'] == 'bad').sum()}")
    df_grp.loc[brand_mask, 'label'] = 'good'

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    df_grp_y = df_grp['label']
    X_cols = df_grp.drop(columns=[
        'label', 'url', 'scheme', 'netloc', 'path',
        'params', 'query', 'fragment', 'domain'
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_cols, df_grp_y, test_size=0.2, random_state=42, stratify=df_grp_y
    )

    numeric_features = ['length', 'domain_hyphens', 'domain_underscores',
                         'path_hyphens', 'path_underscores', 'slashes',
                         'full_stops', 'num_subdomains']
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

    categorical_features = ['tld', 'is_ip']
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    vectorizer_transformer = Pipeline(steps=[
        ('con', Converter()),
        ('tf', TfidfVectorizer())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('domvec', vectorizer_transformer, ['domain_tokens']),
            ('pathvec', vectorizer_transformer, ['path_tokens'])
        ])
    print('model training started')

    # ------------------------------------------------------------------
    # RandomForest with GridSearchCV (primary model - best overfitting control)
    # ------------------------------------------------------------------
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 5]
    }

    # Custom scorer: optimizes for F1 of the 'bad' (phishing) class specifically
    def bad_class_f1(estimator, X, y):
        preds = estimator.predict(X)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        return report.get('bad', {}).get('f1-score', 0.0)

    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=5, n_jobs=-1, scoring=bad_class_f1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best RF parameters: {grid_search.best_params_}")
    rf_clf = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # Other models kept for comparison (pruned DT to reduce overfitting)
    # ------------------------------------------------------------------
    dt_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                                min_samples_split=10, class_weight='balanced',
                                                random_state=42))
    ])
    svc_clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LinearSVC(class_weight='balanced'))])
    log_clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(class_weight='balanced'))])
    nb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', MultinomialNB())])

    dt_clf.fit(X_train, y_train)
    svc_clf.fit(X_train, y_train)
    log_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)

    summary = {}
    summary["Random Forest"] = results("Random Forest", rf_clf, X_train, y_train, X_test, y_test)
    summary["Decision Tree"] = results("Decision Tree", dt_clf, X_train, y_train, X_test, y_test)
    summary["SVC"] = results("SVC", svc_clf, X_train, y_train, X_test, y_test)
    summary["Logistic Regression"] = results("Logistic Regression", log_clf, X_train, y_train, X_test, y_test)
    summary["Naive Bayes"] = results("Naive Bayes", nb_clf, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    # Final summary table across all models (quick side-by-side comparison)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY - Precision / Recall / F1 (phishing='bad' class) by model")
    print("=" * 70)
    print(f"{'Model':<22}{'Train F1':<10}{'Test F1':<10}{'Test Prec':<11}{'Test Rec':<10}{'CV F1':<10}")
    for model_name, m in summary.items():
        cv_str = f"{m['cv_mean']:.3f}" if m['cv_mean'] is not None else "n/a"
        print(f"{model_name:<22}{m['train']['f1_bad']:<10.3f}{m['test']['f1_bad']:<10.3f}"
              f"{m['test']['precision_bad']:<11.3f}{m['test']['recall_bad']:<10.3f}{cv_str:<10}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Threshold tuning on RandomForest to minimize false positives
    # (raise threshold above 0.5 -> fewer legit domains flagged as phishing,
    #  at the cost of catching slightly fewer phishing domains)
    # ------------------------------------------------------------------
    classes = rf_clf.named_steps['classifier'].classes_
    bad_idx = list(classes).index('bad')
    probs = rf_clf.predict_proba(X_test)[:, bad_idx]

    for threshold in [0.5, 0.7, 0.8, 0.9]:
        custom_preds = np.where(probs >= threshold, 'bad', 'good')
        print(f"\n--- RandomForest @ threshold={threshold} ---")
        print(classification_report(y_test, custom_preds, zero_division=0))

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    with open('model/whole_dataset_rf_clf_model.pkl', 'wb') as file:
        pickle.dump(rf_clf, file)
    print("Random Forest model saved as 'whole_dataset_rf_clf_model.pkl'")

    with open('model/whole_dataset_dt_clf_model.pkl', 'wb') as file:
        pickle.dump(dt_clf, file)
    print("Decision Tree model saved as 'whole_dataset_dt_clf_model.pkl'")

    with open('model/whole_dataset_nb_clf_model.pkl', 'wb') as file:
        pickle.dump(nb_clf, file)
    print("nb model saved as 'whole_dataset_nb_clf_model.pkl'")
