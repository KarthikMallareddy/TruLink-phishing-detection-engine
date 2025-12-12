import pandas as pd
import re
import math
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
# We try to load different common filenames
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
    print("❌ Error: No CSV file found in 'datasets/' folder.")
    exit()

# --- 2. CLEAN DATA ---
df.columns = [x.lower() for x in df.columns]

# Handle Labels (Convert 'good'/'bad' to 0/1)
target_col = 'label' if 'label' in df.columns else 'class'
# If the labels are words, map them to numbers
if df[target_col].dtype == 'object':
    print("ℹ️ Converting text labels to numbers...")
    df[target_col] = df[target_col].map({'bad': 1, 'good': 0, 'phishing': 1, 'legitimate': 0})

y = df[target_col]

# --- 3. FEATURE EXTRACTION ---
print("⏳ Extracting features (This takes 1-2 mins)...")

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
    
    # 1. Length
    features.append(len(url))
    # 2. Dots
    features.append(url.count('.'))
    # 3. Slashes
    features.append(url.count('/'))
    # 4. @ Symbol
    features.append(1 if '@' in url else 0)
    # 5. HTTPS
    features.append(1 if 'https' in url else 0)
    
    # 6. Suspicious Words
    sus_words = ['login', 'secure', 'account', 'update', 'verify', 'banking']
    hits = 0
    for word in sus_words:
        if word in url: hits += 1
    features.append(hits)
    
    # 7. Digits
    digits = sum(c.isdigit() for c in url)
    features.append(digits)
    
    # 8. Entropy
    features.append(get_entropy(url))
    
    return features

# Create the "X" variable (The Features)
X = [get_features(url) for url in df['url']]

# --- 4. SPLIT DATA ---
# THIS is the step that was missing or broken before
print("✂️ Splitting data into Train and Test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. TRAIN MODEL ---
print("🚀 Training XGBoost Model...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1)

# Now X_train definitely exists, so this will work
model.fit(X_train, y_train)

# --- 6. SAVE RESULTS ---
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n🎉 Model Accuracy: {acc * 100:.2f}%")

joblib.dump(model, 'phishing_model.pkl')
print("💾 Saved 'phishing_model.pkl'")