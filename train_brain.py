import pandas as pd
import re
import math
import joblib
import scipy.sparse as sp # Needed to glue the two parts together
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
POSSIBLE_FILES = [
    'datasets/phishing_site_urls.csv', 
    'datasets/phishing.csv'
]

df = None
for file in POSSIBLE_FILES:
    try:
        print(f"Loading {file}...")
        df = pd.read_csv(file)
        break
    except FileNotFoundError:
        continue

if df is None:
    print("Error: No CSV found.")
    exit()

# Clean Data
df.columns = [x.lower() for x in df.columns]
target_col = 'label' if 'label' in df.columns else 'class'
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].map({'bad': 1, 'good': 0, 'phishing': 1, 'legitimate': 0})

# Use a smaller sample if your computer freezes (Optional)
# df = df.sample(100000) 

y = df[target_col]
urls = df['url'].astype(str)

# --- 2. PART A: TEXT ANALYSIS (TF-IDF) ---
print("Analyzing URL text patterns (Char-Grams)...")
# analyzer='char': Look at characters, not words
# ngram_range=(3,5): Look at chunks of 3-5 letters (e.g., 'pay', 'ypa', 'pal')
# max_features=5000: Keep only the top 5,000 most common patterns to save memory
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=5000)
X_text = vectorizer.fit_transform(urls)

# --- 3. PART B: MANUAL FEATURES (Your "Forensic" code) ---
print("Extracting manual features...")

def get_entropy(text):
    if not text: return 0
    entropy = 0
    for x in set(text):
        p_x = text.count(x) / len(text)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_manual_features(url):
    features = []
    url = str(url).lower()
    
    # Structural
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('/'))
    features.append(1 if '@' in url else 0)
    features.append(1 if 'https' in url else 0)
    features.append(url.count('-'))
    features.append(url.count('?'))
    features.append(url.count('='))
    
    # Sketchy TLDs
    sketchy = 0
    for tld in ['.xyz', '.top', '.club', '.win', '.info']:
        if url.endswith(tld): sketchy = 1
    features.append(sketchy)
    
    # Suspicious Words
    sus_words = ['login', 'secure', 'account', 'update', 'banking']
    hits = 0
    for word in sus_words:
        if word in url: hits += 1
    features.append(hits)
    
    # Entropy
    features.append(get_entropy(url))
    
    return features

# Generate the list of lists
X_manual_list = [get_manual_features(url) for url in urls]

# Convert to a format that can combine with TF-IDF
X_manual = sp.csr_matrix(X_manual_list)

# --- 4. COMBINE (HYBRID) ---
print("Combining Text + Features...")
# This glues the 5000 text columns with your 11 manual columns
X_final = sp.hstack([X_text, X_manual])

# --- 5. TRAIN ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

print("Training Hybrid XGBoost Model...")
model = XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.1, n_jobs=-1)
model.fit(X_train, y_train)

# --- 6. SAVE ---
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nHYBRID MODEL Accuracy: {acc * 100:.2f}%")

joblib.dump(model, 'hybrid_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl') 
print("Saved 'hybrid_model.pkl' and 'vectorizer.pkl'")