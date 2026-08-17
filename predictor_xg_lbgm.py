import pandas as pd
import pickle
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin
import re
import math
import os
import warnings
import sys
from urllib.parse import urlparse
from functools import lru_cache

warnings.filterwarnings('ignore')


class Converter(BaseEstimator, TransformerMixin):
    """Custom transformer for sklearn pipeline compatibility.
    Must be defined identically here as in train.py - the saved model's
    ColumnTransformer references this class by name when unpickled.
    NOTE: the socket server (phishing_socket_server.py) registers this
    exact class into sys.modules['__main__'] BEFORE importing this module -
    that's what lets pickle.load() succeed when this predictor is imported
    from a different entry-point script."""
    def fit(self, x, y=None):
        return self

    def transform(self, data_frame):
        return data_frame.values.ravel()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


SKIP_PATTERN = re.compile(r'tcp|api|ip6|ip4|dns|mailservice25|cdn', re.IGNORECASE)
INFRA_PATTERN = re.compile(
    r'cdn77|in-addr|inshot|akamaized|akamaihd|cachefly|bitgravity|amazonaws|cloudfront|cdnnetworks|CDNify|'
    r'chinacache|fastly|vivo|netdna|cloudflare|360safe|amazon webservices|CDNsun|azure|azure-api|incapsula|'
    r'limelight networks|gist|amazon|softlayer|ipv4|ipv6|octoshape|cdnvideo|keycdn|staticfile|cdnjs.cn|azureedge|msedge|'
    r'windowsupdate|svc|adsbooster|admaster|docomo|criteo|getapi|gstatic|kotak811|gov|cars24|a2z|outlook|trustedstack|cedexis-radar|navi-tech|'
    r'kingsoft-office-service|reactvision|copilot|wondershare|cip|copiolet|adnxs-simple|tencent-cloud|easy4ipcloud|glance-cdn|v-videoapp|glance-cdn|'
    r'samsung-dict|selectorshub|crowd-umlaut|clevertap-prod|think-cell|we-stats|cc-cluster-2|amazon-adsystem',
    re.IGNORECASE
)
TARGET_PATTERN_6CHAR = re.compile(r'^\d[a-z]\d[a-z]\d[a-z]$')
TARGET_PATTERN_4CHAR = re.compile(r'^[a-z]\d{2}[a-z]$')

_tld_extractor = tldextract.TLDExtract(
    cache_dir='/tmp/tldextract_cache',
    suffix_list_urls=None,
    fallback_to_snapshot=True
)


@lru_cache(maxsize=100000)
def get_tld_extract_cached(url):
    return _tld_extractor(url)


def get_simplified_tld(url):
    extracted = get_tld_extract_cached(url)
    return extracted.suffix if extracted.suffix else None


def get_registered_domain_without_tld(url):
    """Main domain WITHOUT tld, e.g. 'google' from 'www.google.com'."""
    extracted = get_tld_extract_cached(url)
    return extracted.domain.lower() if extracted.domain else ""


def get_full_registered_domain(url):
    """Full registrable domain WITH tld, e.g. 'google.com' from 'www.google.com'.
    Used to match against alexa.pkl, which stores full domains."""
    extracted = get_tld_extract_cached(url)
    if not extracted.domain:
        return ""
    return f"{extracted.domain}.{extracted.suffix}".lower() if extracted.suffix else extracted.domain.lower()


@lru_cache(maxsize=50000)
def check_brand_match(url_lower):
    """Checks the MAIN domain (no TLD) against brand_tokens.pkl.
    IMPORTANT: caches on url_lower ONLY. Do NOT pass the multi-million
    -entry brand set/tuple as an argument here - lru_cache hashes every
    argument to build the cache key, so passing a huge tuple forces
    Python to hash millions of elements on EVERY call (cache hit or
    miss), which is what was causing the 0.6-1.2s slow predictions."""
    try:
        extracted = get_tld_extract_cached(url_lower)
        if extracted.domain and extracted.domain.lower() in brand_name_tokens:
            return True
    except Exception:
        pass
    return False


@lru_cache(maxsize=50000)
def check_alexa_match(url_lower):
    """Checks the FULL domain (with TLD) against alexa.pkl.
    Same fix as check_brand_match - caches on url_lower ONLY, references
    the global alexa_full_domains set directly instead of passing it in."""
    full_domain = get_full_registered_domain(url_lower)
    return bool(full_domain) and full_domain in alexa_full_domains


def looks_like_suspicious_com(domain, brand_tokens):
    ext = get_tld_extract_cached(domain)
    subdomain = ext.subdomain
    reg_domain = ext.domain
    suffix = ext.suffix
    if suffix != "com":
        return False
    if reg_domain in brand_tokens or subdomain in brand_tokens:
        return False
    if not (len(reg_domain) >= 5 and reg_domain.isalnum() and any(c.isdigit() for c in reg_domain)):
        return False
    for part in subdomain.split('.'):
        if len(part) >= 5 and part.isalnum() and any(c.isdigit() for c in part):
            return True
    return False


def looks_like_suspicious_info(url):
    ext = get_tld_extract_cached(url)
    if ext.suffix != "info":
        return False
    domain_part = url.split('/')[0].lower()
    return '.' in domain_part and '-' in domain_part


def calculate_entropy(text):
    if not text:
        return 0
    length = len(text)
    prob = [text.count(c) / length for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def looks_like_suspicious_patterns(url, brand_tokens):
    ext = get_tld_extract_cached(url)
    reg_domain = ext.domain.lower()
    subdomain = ext.subdomain.lower()
    suffix = ext.suffix.lower()
    path = url.split('/', 1)[1] if '/' in url else ""
    if not path and not url.endswith('/'):
        if reg_domain in brand_tokens:
            return False
    suspicious_tlds = {'top', 'wang', 'host', 'buzz', 'fun', 'icu', 'xin', 'shop', 'info'}
    if suffix in suspicious_tlds:
        return True
    if reg_domain and calculate_entropy(reg_domain) > 10.0 and len(reg_domain) >= 8 and any(c.isdigit() for c in reg_domain):
        return True
    if subdomain and calculate_entropy(subdomain) > 10.0 and len(subdomain) >= 8 and any(c.isdigit() for c in subdomain):
        return True
    suspicious_keywords = ['casino']
    if any(keyword in reg_domain or keyword in path.lower() for keyword in suspicious_keywords):
        return True
    if path and (len(path) > 20 or calculate_entropy(path) > 10.0 or sum(c.isupper() for c in path) >= 2 or any(ext in path.lower() for ext in ['.php', '.jp', '.papert'])):
        return True
    return False


def looks_like_target_pattern(domain):
    ext = get_tld_extract_cached(domain)
    subdomain = ext.subdomain
    reg_domain = ext.domain
    suffix = ext.suffix
    if suffix != "com":
        return False
    subdomain_parts = subdomain.split('.')
    target_subdomain = subdomain_parts[0] if subdomain_parts else ""
    if not (len(target_subdomain) == 6 and target_subdomain.isalnum()):
        return False
    pattern_6char = bool(TARGET_PATTERN_6CHAR.match(reg_domain))
    pattern_4char = bool(TARGET_PATTERN_4CHAR.match(reg_domain))
    return pattern_6char or pattern_4char


# ----------------------------------------------------------------------
# Feature helpers - mirror train.py exactly (netloc-only vs path-only
# splits, tldextract-based num_subdomains, letters-only tokenization)
# ----------------------------------------------------------------------
_word_tokenizer = re.compile(r'[A-Za-z]+')


def get_netloc_and_path(query):
    try:
        no_scheme = not query.startswith('https://') and not query.startswith('http://')
        parsed = urlparse(f"http://{query}" if no_scheme else query)
        return parsed.netloc, parsed.path
    except (ValueError, TypeError, AttributeError):
        return "", ""


def get_num_subdomains_matched(netloc):
    try:
        if not netloc:
            return 0
        subdomain = get_tld_extract_cached(netloc).subdomain
        return subdomain.count('.') + 1 if subdomain else 0
    except Exception:
        return 0


def tokenize_domain_matched(netloc):
    try:
        if not netloc:
            return ""
        ext = get_tld_extract_cached(netloc)
        return " ".join(_word_tokenizer.findall(f"{ext.subdomain}.{ext.domain}"))
    except Exception:
        return ""


def tokenize_path_matched(path):
    try:
        return " ".join(_word_tokenizer.findall(path)) if path else ""
    except Exception:
        return ""


def load_domain_set_from_pickle(path, preferred_cols):
    """
    Loads a pickle that may be a DataFrame or a plain set/list, and
    returns a lowercase string set. MUST match train.py's loader exactly:
      - brand_tokens.pkl: DataFrame with a 'brand' column (main domains, no TLD)
      - alexa.pkl: DataFrame with domain values in the FIRST column
        (column header may itself be a domain name if saved without an
        explicit header - the column name is ignored, only values matter)
    """
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


# ----------------------------------------------------------------------
# Load model + supporting data
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print('Loading model...')
model_path = os.path.join(SCRIPT_DIR, 'model', 'whole_dataset_lgbm_clf_model.pkl')
try:
    with open(model_path, 'rb') as f:
        log_model = pickle.load(f)
    print('Model loaded successfully (LightGBM)')
except Exception as e:
    print(f'ERROR loading model: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# train.py encodes labels as integers: {'good': 0, 'bad': 1}. So
# model.classes_ is [0, 1], not ['good','bad'] - class 1 = phishing.
# Threshold=0.9 per verify_output.txt: precision=0.886, recall=0.390.
PHISHING_THRESHOLD = 0.9
BAD_LABEL = 1

print('Loading brand tokens...')
brand_tokens_path = os.path.join(SCRIPT_DIR, 'brand_tokens.pkl')
try:
    brand_name_tokens = load_domain_set_from_pickle(brand_tokens_path, ['brand', 'main_domain', 'domain'])
    brand_name_tokens.add('360safe')
    print(f'Brand tokens loaded successfully ({len(brand_name_tokens)} brands)')
except Exception as e:
    print(f'ERROR loading brand tokens: {e}')
    sys.exit(1)

print('Loading alexa...')
alexa_path = os.path.join(SCRIPT_DIR, 'alexa.pkl')
try:
    alexa_full_domains = load_domain_set_from_pickle(alexa_path, ['domain'])
    print(f'Alexa loaded successfully ({len(alexa_full_domains)} full domains)')
except Exception as e:
    print(f'ERROR loading alexa: {e}')
    sys.exit(1)

trusted_tlds = {
    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'us', 'uk', 'de', 'jp', 'fr', 'au', 'ca',
    'in','com.uy','com.ar', 'ac.jp','com.au','cn','eu', 'br', 'ru', 'online', 'site', 'tech', 'app', 'io', 'ai', 'tv', 'me', 'biz',
    'store','cl','mx','uy','bo', 'blog', 'dev', 'page', 'digital', 'media', 'agency', 'life', 'world','godaddy',
    'news','ms','co.jp', 'software','gov.br', 'live', 'work', 'today', 'cloud', 'academy', 'design', 'law', 'email', 'finance',
    'photography','ac.at', 'video', 'health', 'marketing', 'nic.in', 'co.in', 'ac.in', 'gov.in','edu.au', 'edu.in', 'net.in',
    'org.in','com.sg', 'hk', 'hr', 'ch', 'vn', 'id', 'tel', 'ir', 'monster', 'co.uk', 'games', 'fyi', 'arpa',
    'shop', 'goog', 'technology', 'tw', 'mobi', 'fun', 'com.br', 'com.vn', 'com.hk', 'to', 'microsoft',
    'services', 'aws', 'boo', 'one', 'lc', 'bet', 'vip', 'com.co', 'bid', 'vic.gov.au', 're'
}

suspicious_tlds = {'xyz', 'cc', 'gq', 'wang', 'fun', 'top', 'win', 'icu', 'xin', 'shop', 'buzz'}

trusted_domain_keywords = {
    'google', 'facebook', 'microsoft', 'apple', 'amazon', 'paypal', 'netflix', 'twitter',
    'linkedin', 'github', 'stackoverflow', 'wikipedia', 'reddit', 'ebay', 'instagram', 'yahoo', 'bing'
}


def predict(url: str) -> int:
    try:
        query = url.strip().rstrip('.')
        if query.endswith('.arpa'):
            query = query[:-5].rstrip('.')

        if '://' in query:
            query = query.split('://')[1]

        domain_part = query.split('/')[0].lower() if query else ''
        query_lower = query.lower()

        netloc, path = get_netloc_and_path(query)

        # step 1: STRICT BRAND CHECK - main domain (no TLD) vs brand_tokens.pkl
        if check_brand_match(query_lower):
            return 0

        # step 2: STRICT ALEXA CHECK - full domain (with TLD) vs alexa.pkl
        if check_alexa_match(query_lower):
            return 0

        if SKIP_PATTERN.search(query):
            return 0

        registered_domain = get_registered_domain_without_tld(query) if query else ''
        if registered_domain in trusted_domain_keywords:
            return 0

        domain_tokens_for_check = ' '.join(query.split('/')[0].split('.')) if query else ''
        if INFRA_PATTERN.search(domain_tokens_for_check):
            return 0

        # step 3: Feature extraction and model prediction
        ends_with_slash = query.endswith('/') if query else False
        tld = get_simplified_tld(query)
        tld_exact_match = tld in trusted_tlds and not ends_with_slash if tld else False

        if tld_exact_match:
            label = 'Not Phishing'
        else:
            length = len(query) if query else 0
            is_ip_addr = bool(re.fullmatch(r'\d+\.\d+\.\d+\.\d+', netloc)) if netloc else False
            domain_hyphens = netloc.count('-') if netloc else 0
            domain_underscores = netloc.count('_') if netloc else 0
            path_hyphens = path.count('-') if path else 0
            path_underscores = path.count('_') if path else 0
            slashes = path.count('/') if path else 0
            full_stops_feature = path.count('.') if path else 0
            num_subdomains = get_num_subdomains_matched(netloc)
            domain_tokens = tokenize_domain_matched(netloc)
            path_tokens = tokenize_path_matched(path)
            tld_for_model = tld if tld else 'None'

            df = pd.DataFrame({
                'length': [length],
                'tld': [tld_for_model],
                'is_ip': [is_ip_addr],
                'domain_hyphens': [domain_hyphens],
                'domain_underscores': [domain_underscores],
                'path_hyphens': [path_hyphens],
                'path_underscores': [path_underscores],
                'slashes': [slashes],
                'full_stops': [full_stops_feature],
                'num_subdomains': [num_subdomains],
                'domain_tokens': [domain_tokens],
                'path_tokens': [path_tokens]
            })

            X = df[[
                'length', 'tld', 'is_ip', 'domain_hyphens', 'domain_underscores',
                'path_hyphens', 'path_underscores', 'slashes', 'full_stops',
                'num_subdomains', 'domain_tokens', 'path_tokens'
            ]]

            classes = log_model.named_steps['classifier'].classes_
            bad_idx = list(classes).index(BAD_LABEL)
            prob_bad = log_model.predict_proba(X)[0][bad_idx]

            label = 'Phishing' if prob_bad >= PHISHING_THRESHOLD else 'Not Phishing'

        # step 4: heuristic checks (reuse netloc/path, consistent with above)
        has_digit = any(char.isdigit() for char in query) if query else False
        full_stops = path.count('.') if path else 0
        domain_hyphens = netloc.count('-') if netloc else 0

        if not has_digit and full_stops < 3 and domain_hyphens < 3:
            label = 'Not Phishing'

        if tld in suspicious_tlds:
            label = 'Phishing'

        if looks_like_suspicious_com(domain_part, brand_name_tokens):
            label = 'Phishing'

        if looks_like_suspicious_info(query):
            label = 'Phishing'

        if looks_like_target_pattern(domain_part):
            label = 'Phishing'

        if looks_like_suspicious_patterns(query, brand_name_tokens):
            label = 'Phishing'

        return 1 if label == 'Phishing' else 0

    except Exception as e:
        print(f"Error in prediction for URL '{url}': {e}", file=sys.stderr)
        return 0
