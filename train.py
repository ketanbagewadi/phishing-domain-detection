import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix
import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics
import plotly.express as px
import time # calculate time
from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad
from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos
import pickle# use to dump model
import warnings # ignores pink warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings("ignore")
from urllib.parse import urlparse
import re
import tldextract
from typing import Dict, List, Optional
from urllib.parse import urlparse



def extract_features(url):
    try:
        # Handle NaN or non-string values
        if pd.isna(url) or not isinstance(url, str):
            return {
                'length_url': 0,
                'length_hostname': 0,
                'ip': 0,
                'nb_dots': 0,
                'nb_hyphens': 0,
                'nb_at': 0
            }
        
        parsed_url = urlparse(url)
        features = {}
         
        # Basic features
        features['length_url'] = len(url)
        features['length_hostname'] = len(parsed_url.hostname) if parsed_url.hostname else 0
        features['ip'] = 1 if parsed_url.hostname and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", parsed_url.hostname) else 0
        features['nb_dots'] = parsed_url.hostname.count('.') if parsed_url.hostname else 0
        features['nb_hyphens'] = parsed_url.hostname.count('-') if parsed_url.hostname else 0
        features['nb_at'] = url.count('@')
        return features
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error parsing URL '{url}': {e}")
        # Return default values for problematic URLs
        return {
            'length_url': len(str(url)) if url else 0,
            'length_hostname': 0,
            'ip': 0,
            'nb_dots': 0,
            'nb_hyphens': 0,
            'nb_at': str(url).count('@') if url else 0
        }


def parse_url(url: str) -> Optional[Dict[str, str]]:
    try:
        # Handle NaN or non-string values
        if pd.isna(url) or not isinstance(url, str):
            return {
                "scheme": None, 
                "netloc": None,
                "path": "",
                "params": "",
                "query": "",
                "fragment": "",
            }
            
        # Clean the URL - remove extra spaces and handle common issues
        url = str(url).strip()
        if not url:
            return {
                "scheme": None, 
                "netloc": None,
                "path": "",
                "params": "",
                "query": "",
                "fragment": "",
            }
        
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "scheme": None, 
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except (ValueError, TypeError, AttributeError) as e:
        print(f"An error occurred while parsing the URL '{url}': {e}")
        return {
            "scheme": None, 
            "netloc": None,
            "path": "",
            "params": "",
            "query": "",
            "fragment": "",
        }


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

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
def tokenize_domain(netloc: str) -> str:
    try:
        if pd.isna(netloc) or not isinstance(netloc, str) or not netloc.strip():
            return ""
        split_domain = tldextract.extract(netloc)
        no_tld = str(split_domain.subdomain +'.'+ split_domain.domain)
        return " ".join(map(str,tokenizer.tokenize(no_tld)))
    except Exception as e:
        print(f"Error tokenizing domain '{netloc}': {e}")
        return ""


class Converter(BaseEstimator, TransformerMixin):   
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()


def results(name: str, model: BaseEstimator) -> None:
    preds = model.predict(X_test)

    print(name + " score: %.3f" % model.score(X_test, y_test))
    print(classification_report(y_test, preds))
    labels = ['good', 'bad']

    conf_matrix = confusion_matrix(y_test, preds)

    font = {'family' : 'normal',
            'size'   : 14}

    plt.rc('font', **font)
    plt.figure(figsize= (10,6))
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap='Greens')
    plt.title("Confusion Matrix for " + name)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('all_urls.csv')
    print(f"Loaded dataset with {len(df)} rows")
    
    # Clean the data - remove rows with missing URLs
    df = df.dropna(subset=['url'])
    print(f"After removing missing URLs: {len(df)} rows")
    
    # Convert URLs to string type to handle any type issues
    df['url'] = df['url'].astype(str)
    
    df.head() 

    # Convert numeric labels to text labels
    # 0 -> good, 1 -> bad/phishing
    df['label'] = df['label'].map({0: 'good', 1: 'bad'})
    
    print("Label distribution:")
    print(df['label'].value_counts())

    df['features'] = df['url'].apply(extract_features)


    df = pd.concat([df.drop(['features'], axis=1), df['features'].apply(pd.Series)], axis=1)

    # Display the DataFrame with extracted features
    print(df)


    # Group by URL and combine labels into a list
    df_grp = df.groupby("url")["label"].agg(list).reset_index()

    # Apply the parse_url function to the "url" column and store the results in a new column "parsed_url"
    df_grp["parsed_url"] = df_grp["url"].apply(parse_url)

    # Combine the labels into a single string 
    df_grp["label"] = df_grp["label"].apply(lambda labels: ''.join(map(str, labels)))

    # first character of the label
    df_grp["label"] = df_grp["label"].apply(lambda x: x[0] if x else '')

    # lowercase formatting
    df_grp['label'] = df_grp['label'].str.lower()

    # Replace abbreviations (now handles 'g' for good and 'b' for bad)
    df_grp['label'] = df_grp['label'].replace({'g': 'good', 'b': 'bad'})



    # Display the DataFrame
    print(df_grp.head())


    df_grp = pd.concat([
        df_grp.drop(['parsed_url'], axis=1),
        df_grp['parsed_url'].apply(pd.Series)
    ], axis=1)
    df_grp

    # Remove rows where 'netloc' is null or empty after parsing
    df_grp = df_grp.dropna(subset=['netloc'])
    df_grp = df_grp[df_grp['netloc'] != '']
    df_grp = df_grp[df_grp['netloc'].notna()]
    print(f"After removing invalid netloc: {len(df_grp)} rows")


    # Apply the function to extract the TLD for each URL in the 'netloc' column
    df_grp["tld"] = df_grp.netloc.apply(extract_tld)

    # Replace any empty TLDs with 'None'
    df_grp['tld'] = df_grp['tld'].replace('', 'None')
    df_grp["length"] = df_grp.url.str.len()

    #The TLD is then extracted using a python library, and if no TLD is present simply add 'None'.
    df_grp["tld"] = df_grp.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
    df_grp['tld'] = df_grp['tld'].replace('','None')
    df_grp["is_ip"] = df_grp.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")

    # Make sure that df_grp and its columns exist and are correctly named
    df_grp['domain_hyphens'] = df_grp.netloc.str.count('-')
    df_grp['domain_underscores'] = df_grp.netloc.str.count('_')
    df_grp['path_hyphens'] = df_grp.path.fillna('').str.count('-')
    df_grp['path_underscores'] = df_grp.path.fillna('').str.count('_')
    df_grp['slashes'] = df_grp.path.fillna('').str.count('/')
    df_grp['full_stops'] = df_grp.path.fillna('').str.count('.')
    df_grp['num_subdomains'] = df_grp['netloc'].apply(lambda net: get_num_subdomains(net))
    df_grp['domain_tokens'] = df_grp['netloc'].apply(lambda net: tokenize_domain(net))
    df_grp['path_tokens'] = df_grp['path'].fillna('').apply(lambda path: " ".join(map(str,tokenizer.tokenize(path))))
    df_grp.columns.tolist()

    df_grp_y = df_grp['label'] 
    df_grp.drop('label', axis=1, inplace=True) 
    df_grp.drop('url', axis=1, inplace=True)
    df_grp.drop('scheme', axis=1, inplace=True)
    df_grp.drop('netloc', axis=1, inplace=True)
    df_grp.drop('path', axis=1, inplace=True)
    df_grp.drop('params', axis=1, inplace=True)
    df_grp.drop('query', axis=1, inplace=True)
    df_grp.drop('fragment', axis=1, inplace=True)
    df_grp

    X_train, X_test, y_train, y_test = train_test_split(df_grp, df_grp_y, test_size=0.2)

    numeric_features = ['length', 'domain_hyphens', 'domain_underscores', 'path_hyphens', 'path_underscores', 'slashes', 'full_stops', 'num_subdomains']
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    categorical_features = ['tld', 'is_ip']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    vectorizer_features = ['domain_tokens','path_tokens']
    vectorizer_transformer = Pipeline(steps=[
        ('con', Converter()),
        ('tf', TfidfVectorizer())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('domvec', vectorizer_transformer, ['domain_tokens']),
            ('pathvec', vectorizer_transformer, ['path_tokens'])
        ])
    print(' model started')

    from sklearn.tree import DecisionTreeClassifier

# Define Decision Tree pipeline
dt_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', DecisionTreeClassifier())])

# Train the Decision Tree model
dt_clf.fit(X_train, y_train)

# Evaluate the Decision Tree model
results("Decision Tree", dt_clf)

svc_clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LinearSVC())])

log_clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])
    
nb_clf = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', MultinomialNB())])
svc_clf.fit(X_train, y_train)
log_clf.fit(X_train, y_train)
nb_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)

results("SVC" , svc_clf)
results("Logistic Regression" , log_clf)
results("Naive Bayes" , nb_clf)
results(" DecisionTreeClassifier" , dt_clf)


import pickle

# Save the Decision Tree model
with open('model/whole_dataset_dt_clf_model.pkl', 'wb') as file:
    pickle.dump(dt_clf, file)
print("Decision Tree model saved as 'dt_clf_model.pkl'")

# Load the Decision Tree model
with open('model/whole_dataset_dt_clf_model.pkl', 'rb') as file:
    dt_clf_loaded = pickle.load(file)
print("Decision Tree model loaded successfully!")

#save the NB model
with open('model/whole_dataset_nb_clf_model.pkl','wb') as file:
    pickle.dump(nb_clf,file)
print("nb model saved as 'nb_clf_model.pkl'")


#load the nb model
with open('model/whole_dataset_nb_clf_model.pkl','rb') as file:
    nb_clf_loaded = pickle.load(file)
print('nb model loaded succssfully')
