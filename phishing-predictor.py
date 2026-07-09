import os
import sys
import math
import re
import time
import argparse
import pickle
import ipaddress
from functools import lru_cache

import pandas as pd
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')


class Converter(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, data_frame):
        return data_frame.values.ravel()
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        pass


# Pre-compiled regex patterns
SKIP_PATTERN = re.compile(r'tcp|api|ip6|ip4|dns|mailservice25|cdn', re.IGNORECASE)
INFRA_PATTERN = re.compile(
    r'cdn77|in-addr|akamaized|akamaihd|cachefly|bitgravity|amazonaws|cloudfront|cdnnetworks|CDNify|'
    r'chinacache|fastly|vivo|netdna|cloudflare|360safe|amazon webservices|CDNsun|azure|azure-api|incapsula|'
    r'limelight networks|gist|amazon|softlayer|ipv4|ipv6|octoshape|cdnvideo|keycdn|staticfile|cdnjs.cn|azureedge|msedge|'
    r'windowsupdate|svc|docomo|criteo|getapi|gstatic|kotak811|gov|cars24|a2z|outlook|trustedstack|cedexis-radar|navi-tech|'
    r'kingsoft-office-service|adnxs-simple|tencent-cloud|easy4ipcloud|glance-cdn|v-videoapp|glance-cdn|'
    r'samsung-dict|crowd-umlaut|clevertap-prod|think-cell|we-stats|cc-cluster-2|amazon-adsystem',
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


def is_ip(url):
    try:
        ipaddress.IPv4Address(url.split('/')[0])
        return True
    except ipaddress.AddressValueError:
        try:
            ipaddress.IPv6Address(url.split('/')[0])
            return True
        except ipaddress.AddressValueError:
            return False


def get_registered_domain_without_tld(url):
    extracted = get_tld_extract_cached(url)
    return extracted.domain.lower() if extracted.domain else ""


def calculate_entropy(text):
    if not text:
        return 0
    length = len(text)
    prob = [text.count(c) / length for c in set(text)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def looks_like_suspicious_com(host, brand_set):
    ext        = get_tld_extract_cached(host)
    subdomain  = ext.subdomain.lower()
    reg_domain = ext.domain.lower()
    suffix     = ext.suffix.lower()

    if suffix != "com":
        return False
    if reg_domain in brand_set or subdomain in brand_set:
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


def looks_like_suspicious_patterns(url, brand_set):
    ext        = get_tld_extract_cached(url)
    reg_domain = ext.domain.lower()
    subdomain  = ext.subdomain.lower()
    suffix     = ext.suffix.lower()
    path       = url.split('/', 1)[1] if '/' in url else ""

    if reg_domain in brand_set:
        return False

    suspicious_tlds = {'top','wang','host','buzz', 'fun', 'icu', 'xin', 'shop', 'info'}
    if suffix in suspicious_tlds:
        return True

    if reg_domain and calculate_entropy(reg_domain) > 10.0 and len(reg_domain) >= 8 and any(c.isdigit() for c in reg_domain):
        return True
    if subdomain and calculate_entropy(subdomain) > 10.0 and len(subdomain) >= 8 and any(c.isdigit() for c in subdomain):
        return True

    suspicious_keywords = ['casino']
    if any(keyword in reg_domain or keyword in path.lower() for keyword in suspicious_keywords):
        return True

    if path and (
        len(path) > 20
        or calculate_entropy(path) > 10.0
        or sum(c.isupper() for c in path) >= 2
        or any(e in path.lower() for e in ['.php', '.jp', '.papert'])
    ):
        return True

    return False


def looks_like_target_pattern(host):
    ext        = get_tld_extract_cached(host)
    subdomain  = ext.subdomain
    reg_domain = ext.domain
    suffix     = ext.suffix

    if suffix != "com":
        return False

    subdomain_parts  = subdomain.split('.')
    target_subdomain = subdomain_parts[0] if subdomain_parts else ""

    if not (len(target_subdomain) == 6 and target_subdomain.isalnum()):
        return False

    return bool(TARGET_PATTERN_6CHAR.match(reg_domain)) or bool(TARGET_PATTERN_4CHAR.match(reg_domain))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_asset(path, label):
    print(f"Loading {label} …")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"   {label} loaded")
        return obj
    except Exception as e:
        print(f"   ERROR loading {label}: {e}")
        sys.exit(1)


log_model = _load_asset(
    os.path.join(SCRIPT_DIR, 'model', 'whole_dataset_dt_clf_model.pkl'), 'model')

_brand_raw = _load_asset(
    os.path.join(SCRIPT_DIR, 'brand_tokens.pkl'), 'brand tokens')
_brand_raw.add('360safe')
BRAND_SET = {str(t).strip().lower() for t in _brand_raw if t}
print(f"  Brand entries: {len(BRAND_SET)}")

alexa_tlds = _load_asset(os.path.join(SCRIPT_DIR, 'alexa.pkl'), 'alexa')
if not isinstance(alexa_tlds, set):
    alexa_tlds = set(alexa_tlds)

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

suspicious_tlds = {'xyz', 'cc','gq','wang', 'fun', 'top', 'win', 'icu', 'xin', 'shop', 'buzz'}

trusted_domain_keywords = {
    'google', 'facebook', 'microsoft', 'apple', 'amazon', 'paypal', 'netflix', 'twitter',
    'linkedin', 'github', 'stackoverflow', 'wikipedia', 'reddit', 'ebay', 'instagram', 'yahoo', 'bing'
}


def predict(url: str) -> int:
    """Returns 0 (not phishing) or 1 (phishing)."""
    try:
        query = url.strip().rstrip('.')
        if query.endswith('.arpa'):
            query = query[:-5].rstrip('.')
        if '://' in query:
            query = query.split('://')[1]
        if not query:
            return 0

        query_lower = query.lower()
        host        = query_lower.split('/')[0]

        # Priority 1 – brand bypass
        parsed     = get_tld_extract_cached(host)
        reg_domain = parsed.domain.strip().lower() if parsed.domain else ""
        if reg_domain and reg_domain in BRAND_SET:
            return 0

        # Priority 2 – quick whitelists
        simple_suffix = host.split('.')[-1] if '.' in host else ''
        if simple_suffix in alexa_tlds:
            return 0
        if SKIP_PATTERN.search(query_lower):
            return 0
        tld_check = get_simplified_tld(host)
        if tld_check and tld_check in alexa_tlds:
            return 0
        if reg_domain in trusted_domain_keywords:
            return 0
        host_tokens = ' '.join(host.split('.'))
        if INFRA_PATTERN.search(host_tokens):
            return 0

        # Priority 3 – ML model
        tld             = tld_check
        ends_with_slash = query.endswith('/')
        tld_exact_match = (tld in trusted_tlds and not ends_with_slash) if tld else False

        if tld_exact_match:
            label = 'Not Phishing'
        else:
            path = query.split('/', 1)[1] if '/' in query else ''
            df   = pd.DataFrame({
                'url':                [query],
                'length':             [len(query)],
                'tld':                [tld],
                'is_ip':              [is_ip(query)],
                'domain_hyphens':     [query.count('-')],
                'domain_underscores': [query.count('_')],
                'path_hyphens':       [path.count('-')],
                'path_underscores':   [path.count('_')],
                'slashes':            [query.count('/')],
                'full_stops':         [query.count('.')],
                'num_subdomains':     [query.count('.') - 1 if '.' in query else 0],
                'domain_tokens':      [' '.join(host.split('.'))],
                'path_tokens':        [' '.join(query.split('/')[1:]) if '/' in query else ''],
            })
            X     = df[['length','tld','is_ip','domain_hyphens','domain_underscores',
                         'path_hyphens','path_underscores','slashes','full_stops',
                         'num_subdomains','domain_tokens','path_tokens']]
            pred  = log_model.predict(X)[0]
            label = 'Phishing' if pred == 'bad' else 'Not Phishing'

        # Priority 4 – heuristics
        has_digit      = any(c.isdigit() for c in query)
        full_stops     = query.count('.')
        domain_hyphens = query.count('-')

        if not has_digit and full_stops < 3 and domain_hyphens < 3:
            label = 'Not Phishing'
        if tld in suspicious_tlds:
            label = 'Phishing'
        if looks_like_suspicious_com(host, BRAND_SET):
            label = 'Phishing'
        if looks_like_suspicious_info(query_lower):
            label = 'Phishing'
        if looks_like_target_pattern(host):
            label = 'Phishing'
        if looks_like_suspicious_patterns(query_lower, BRAND_SET):
            label = 'Phishing'

        return 1 if label == 'Phishing' else 0

    except Exception as e:
        print(f"Error predicting '{url}': {e}", file=sys.stderr)
        return 0


def _fmt_duration(seconds: float) -> str:
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {s:.1f}s"
    if seconds >= 1:
        return f"{seconds:.3f}s"
    return f"{seconds * 1000:.3f}ms"


def _run_prediction_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a 'query' column:
      1. Normalise and deduplicate queries.
      2. Predict each unique query exactly once.
      3. Return a slim DataFrame with only:
             query | phishing_flag | phishing_label
    """
    queries_norm    = df['query'].astype(str).str.strip().str.lower().tolist()
    before_dedup    = len(queries_norm)

    seen            = set()
    unique_queries  = []
    for q in queries_norm:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    unique_count    = len(unique_queries)
    duplicate_count = before_dedup - unique_count

    print(f"\n   Query stats:")
    print(f"     Total rows (with duplicates) : {before_dedup:,}")
    if before_dedup:
        print(f"     Duplicate rows removed       : {duplicate_count:,}  "
              f"({duplicate_count / before_dedup * 100:.1f}% of total)")
    print(f"     Unique queries to evaluate   : {unique_count:,}")
    print(f"\n   Running phishing detection on {unique_count:,} unique queries …\n")

    flags       = []
    query_times = []
    pipeline_start = time.perf_counter()
    pad = len(str(unique_count)) if unique_count else 1

    for i, q in enumerate(unique_queries, 1):
        t0      = time.perf_counter()
        flag    = predict(q)
        elapsed = time.perf_counter() - t0
        flags.append(flag)
        query_times.append(elapsed)

        if i % 500 == 0 or i == unique_count:
            pct = i / unique_count * 100
            print(f"     [{i:>{pad}}/{unique_count}]  {pct:5.1f}%  "
                  f"last query: {_fmt_duration(elapsed)}")

    total_elapsed      = time.perf_counter() - pipeline_start
    avg_elapsed        = sum(query_times) / len(query_times) if query_times else 0.0
    throughput         = unique_count / total_elapsed if total_elapsed > 0 else 0.0
    total_phishing     = sum(flags)
    total_not_phishing = unique_count - total_phishing

    print(f"\n   Phishing     : {total_phishing:,}")
    print(f"   Not Phishing : {total_not_phishing:,}")
    print(f"   Avg / query  : {_fmt_duration(avg_elapsed)}")
    print(f"   Total time   : {_fmt_duration(total_elapsed)}")
    print(f"   Throughput   : {throughput:,.1f} queries/sec")

    return pd.DataFrame({
        'query':          unique_queries,
        'phishing_flag':  flags,
        'phishing_label': ['Phishing' if f == 1 else 'Not Phishing'
                           for f in flags],
    })


def process_csv(input_path: str, output_path: str, column: str = None) -> None:
    """
    Read domains/URLs from a CSV file, run phishing detection, write results to a CSV.

    input_path : path to the input CSV containing the domains/URLs to test
    output_path: path to write the results CSV to
    column     : name of the column holding the domain/URL. If not given, the
                 script auto-detects 'query' or 'domain' (case-insensitive).
    """
    print(f"\n Reading input file: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"    Could not read input CSV: {e}")
        sys.exit(1)

    df.columns = [c.strip() for c in df.columns]

    if column:
        if column not in df.columns:
            print(f"    Column '{column}' not found. Columns available: {list(df.columns)}")
            sys.exit(1)
        col = column
    else:
        lower_map = {c.lower(): c for c in df.columns}
        col = lower_map.get('query') or lower_map.get('domain') or lower_map.get('url')
        if not col:
            print(f"    Could not auto-detect a domain/query column. "
                  f"Columns found: {list(df.columns)}. Use --column to specify it.")
            sys.exit(1)

    print(f"   Using column '{col}' as the domain/query field")
    print(f"   Rows loaded: {len(df):,}")

    working = df[[col]].rename(columns={col: 'query'})
    result  = _run_prediction_on_df(working)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\n   Result saved to : {output_path}")
    print("═" * 62)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run phishing prediction on a CSV of domains/URLs and write results to a CSV."
    )
    parser.add_argument('input_csv', help="Path to input CSV file containing domains/URLs")
    parser.add_argument(
        '-o', '--output',
        default=None,
        help="Path to output CSV file (default: <input_name>_results.csv)"
    )
    parser.add_argument(
        '-c', '--column',
        default=None,
        help="Name of the column containing the domain/URL (auto-detected if omitted)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("═" * 62)
    print("  DNS PHISHING DETECTION — CSV MODE")
    print("═" * 62 + "\n")

    args = parse_args()

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.input_csv)
        output_path = f"{base}_results.csv"

    process_csv(args.input_csv, output_path, column=args.column)
