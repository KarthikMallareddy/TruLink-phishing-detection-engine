import pandas as pd
import re
import math
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
POSSIBLE_FILES = [
    'datasets/phishing_site_urls.csv', 
    'datasets/phishing.csv', 
    'datasets/data.csv'
]

df = None
for file in POSSIBLE_FILES:
    try:
        print(f"🔍 Trying to load {file}...")
        df = pd.read_csv(file)
        print(f"✅ Success! Loaded {file}")
        break
    except FileNotFoundError:
        continue

if df is None:
    print("❌ Error: No CSV file found. Please check your folder.")
    exit()

# --- 2. CLEAN DATA ---
df.columns = [x.lower() for x in df.columns]
target_col = 'label' if 'label' in df.columns else 'class'

# Map text labels to numbers if needed
if df[target_col].dtype == 'object':
    print("ℹ️ Mapping labels...")
    df[target_col] = df[target_col].map({'bad': 1, 'good': 0, 'phishing': 1, 'legitimate': 0})

y = df[target_col]

# --- 3. ADVANCED FEATURE EXTRACTION ---
print("⏳ Extracting 'Forensic' features (This takes 2-3 mins)...")

def get_entropy(text):
    if not text: return 0
    entropy = 0
    for x in set(text):
        p_x = text.count(x) / len(text)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_features(url):
    features = []
    url = str(url).lower()
    
    # --- Structural Features ---
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('/'))
    features.append(1 if '@' in url else 0)
    features.append(1 if 'https' in url else 0)
    
    # --- Forensic Features (The Accuracy Boosters) ---
    
    # 1. Hyphens (Crucial for "typosquatting" like face-book.com)
    features.append(url.count('-'))
    
    # 2. Query Parameters (Scams often use ?redirect=...)
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('_'))
    
    # 3. Sketchy TLD Check
    # Legit sites usually are .com, .org, .net. Scams use cheap ones.
    sketchy_tlds = ['.xyz', '.top', '.club', '.win', '.info', '.gq', '.tk', '.ml']
    is_sketchy = 0
    for tld in sketchy_tlds:
        if url.endswith(tld):
            is_sketchy = 1
    features.append(is_sketchy)
    
    # 4. Suspicious Words
    sus_words = ['login', 'secure', 'account', 'update', 'verify', 'banking', 'confirm', 'wallet']
    hits = 0
    for word in sus_words:
        if word in url: hits += 1
    features.append(hits)
    
    # 5. Digit Ratio (Legit sites have few numbers, scams have many)
    digits = sum(c.isdigit() for c in url)
    features.append(digits)
    features.append(digits / len(url) if len(url) > 0 else 0)
    
    # 6. Entropy
    features.append(get_entropy(url))
    
    return features

X = [get_features(url) for url in df['url']]

# --- 4. SPLIT DATA ---
print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. TRAIN MODEL (Aggressive Tuning) ---
print("🚀 Training High-Performance XGBoost...")
# n_estimators=1000: Try 1000 times to improve.
# max_depth=12: Look deeper into data complexity.
model = XGBClassifier(n_estimators=1000, max_depth=12, learning_rate=0.05, n_jobs=-1)

model.fit(X_train, y_train)

# --- 6. SAVE RESULTS ---
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n🔥 SUPERCHARGED Accuracy: {acc * 100:.2f}%")

joblib.dump(model, 'phishing_model.pkl')
print("💾 Saved 'phishing_model.pkl'")