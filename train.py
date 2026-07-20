import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import tldextract
from typing import Dict, List, Optional



# Feature helpers (these MUST match what predictor.py computes at inference)

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
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error parsing URL '{url}': {e}")
        return {'length_url': len(str(url)) if url else 0, 'length_hostname': 0, 'ip': 0,
                 'nb_dots': 0, 'nb_hyphens': 0, 'nb_at': str(url).count('@') if url else 0}


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
    except (ValueError, TypeError, AttributeError) as e:
        print(f"An error occurred while parsing the URL '{url}': {e}")
        return {"scheme": None, "netloc": None, "path": "", "params": "", "query": "", "fragment": ""}


def combine_labels(labels: List[str]) -> str:
    return ''.join(labels)


def extract_tld(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return 'None'
        return tldextract.extract(netloc).suffix
    except Exception as e:
        print(f"Error extracting TLD from '{netloc}': {e}")
        return 'None'


def get_num_subdomains(netloc: str) -> int:
    """Matches predictor.py's need: count of subdomain labels via tldextract, NOT raw dot-count of the URL."""
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return 0
        subdomain = tldextract.extract(netloc).subdomain
        if subdomain == "":
            return 0
        return subdomain.count('.') + 1
    except Exception as e:
        print(f"Error getting subdomains from '{netloc}': {e}")
        return 0


def get_registered_domain(netloc: str) -> str:
    """Registered domain without TLD, e.g. 'paypal' from 'login.paypal.com'."""
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        return tldextract.extract(netloc).domain.lower()
    except Exception:
        return ""


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


def tokenize_domain(netloc: str) -> str:
    """Letters-only tokens from subdomain+domain -- matches predictor.py's domain_tokens expectation."""
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        split_domain = tldextract.extract(netloc)
        no_tld = str(split_domain.subdomain + '.' + split_domain.domain)
        return " ".join(map(str, tokenizer.tokenize(no_tld)))
    except Exception as e:
        print(f"Error tokenizing domain '{netloc}': {e}")
        return ""


def tokenize_path(path: str) -> str:
    """Letters-only tokens from the path -- matches predictor.py's path_tokens expectation."""
    try:
        if pd.isna(path) or not isinstance(path, str):
            return ""
        return " ".join(map(str, tokenizer.tokenize(path)))
    except Exception as e:
        print(f"Error tokenizing path '{path}': {e}")
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


def results(name: str, model: BaseEstimator) -> None:
    preds = model.predict(X_test)
    print(name + " score: %.3f" % model.score(X_test, y_test))
    print(classification_report(y_test, preds))
    labels = ['good', 'bad']
    conf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Greens')
    plt.title("Confusion Matrix for " + name)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load & clean data

    df = pd.read_csv('all_urls.csv')
    print(f"Loaded dataset with {len(df)} rows")

    df = df.dropna(subset=['url'])
    print(f"After removing missing URLs: {len(df)} rows")

    df['url'] = df['url'].astype(str)
    df['label'] = df['label'].map({0: 'good', 1: 'bad'})
    print("Label distribution:")
    print(df['label'].value_counts())

    # ------------------------------------------------------------------
    # Group / parse

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
    # Feature engineering (netloc-only vs path-only, matching predictor.py)
    # ------------------------------------------------------------------
    df_grp["tld"] = df_grp.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
    df_grp['tld'] = df_grp['tld'].replace('', 'None')
    df_grp["length"] = df_grp.url.str.len()
    df_grp["is_ip"] = df_grp.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")

    # domain-side counts come ONLY from netloc
    df_grp['domain_hyphens'] = df_grp.netloc.str.count('-')
    df_grp['domain_underscores'] = df_grp.netloc.str.count('_')

    # path-side counts come ONLY from path
    df_grp['path_hyphens'] = df_grp.path.fillna('').str.count('-')
    df_grp['path_underscores'] = df_grp.path.fillna('').str.count('_')
    df_grp['slashes'] = df_grp.path.fillna('').str.count('/')
    df_grp['full_stops'] = df_grp.path.fillna('').str.count('.')

    df_grp['num_subdomains'] = df_grp['netloc'].apply(get_num_subdomains)
    df_grp['domain'] = df_grp['netloc'].apply(get_registered_domain)
    df_grp['domain_tokens'] = df_grp['netloc'].apply(tokenize_domain)
    df_grp['path_tokens'] = df_grp['path'].fillna('').apply(tokenize_path)

    # ------------------------------------------------------------------
    # Build & save brand_tokens.pkl as a flat SET of domain names
    # (predictor.py does: brand_name_tokens.add('360safe') and
    #  tuple(sorted(brand_name_tokens)) -- it needs a set/list of strings,
    #  NOT a DataFrame)
    # ------------------------------------------------------------------
    MIN_BRAND_LEN = 3
    brand_domain_counts = (
        df_grp[df_grp['label'] == 'good']['domain']
        .value_counts()
    )
    brand_name_tokens = set(
        brand_domain_counts[
            (brand_domain_counts.index.str.len() >= MIN_BRAND_LEN) &
            (brand_domain_counts.index != '')
        ].index
    )
    print(f"Built {len(brand_name_tokens)} brand tokens")

    with open('brand_tokens.pkl', 'wb') as f:
        pickle.dump(brand_name_tokens, f)
    print("brand_tokens.pkl saved as a set of domain strings")

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    df_grp_y = df_grp['label']
    X_cols = df_grp.drop(columns=[
        'label', 'url', 'scheme', 'netloc', 'path',
        'params', 'query', 'fragment', 'domain'
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_cols, df_grp_y, test_size=0.2)

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
    print('model started')

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    dt_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())])
    svc_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearSVC())])
    log_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
    nb_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MultinomialNB())])

    dt_clf.fit(X_train, y_train)
    svc_clf.fit(X_train, y_train)
    log_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)

    results("Decision Tree", dt_clf)
    results("SVC", svc_clf)
    results("Logistic Regression", log_clf)
    results("Naive Bayes", nb_clf)

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    with open('model/whole_dataset_dt_clf_model.pkl', 'wb') as file:
        pickle.dump(dt_clf, file)
    print("Decision Tree model saved as 'whole_dataset_dt_clf_model.pkl'")

    with open('model/whole_dataset_nb_clf_model.pkl', 'wb') as file:
        pickle.dump(nb_clf, file)
    print("nb model saved as 'whole_dataset_nb_clf_model.pkl'")
