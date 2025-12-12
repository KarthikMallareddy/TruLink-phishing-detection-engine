import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIGURATION ---
# We try to load different common filenames in case you renamed it
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
    print("Please download the dataset and place it in PhishingProject/datasets/")
    exit()

# --- 2. PREPARE DATA ---
# Standardize column names
df.columns = [x.lower() for x in df.columns]

# Check for correct columns
if 'url' not in df.columns:
    print("❌ Error: Your CSV must have a 'url' column.")
    print(f"Found columns: {df.columns}")
    exit()

# Handle Labels (Some datasets use 'good/bad', some use '0/1')
target_col = 'label' if 'label' in df.columns else 'class'
print(f"ℹ️ Using '{target_col}' as the target label.")

# Convert 'good'/'bad' to 0/1 if necessary
y = df[target_col]
if y.dtype == 'object':
    y = y.map({'bad': 1, 'good': 0, 'phishing': 1, 'legitimate': 0})

# --- 3. FEATURE EXTRACTION (The "Secret Sauce") ---
print("⏳ Extracting features from URLs (This will take 1-2 mins)...")

import math
from collections import Counter

def get_entropy(text):
    # Measures how "random" a string looks
    if not text:
        return 0
    entropy = 0
    for x in text:
        p_x = text.count(x) / len(text)
        entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_features(url):
    features = []
    url = str(url).lower()
    
    # --- BASICS ---
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('/'))
    features.append(1 if '@' in url else 0)
    features.append(1 if 'https' in url else 0)
    
    # --- ADVANCED CLUES ---
    
    # 1. Suspicious Keyword Check
    # Scammers love these words. Legit sites usually don't put them in the URL path.
    sus_words = ['login', 'secure', 'account', 'update', 'verify', 'banking', 'signin', 'confirm']
    hits = 0
    for word in sus_words:
        if word in url:
            hits += 1
    features.append(hits)
    
    # 2. Count Digits
    # Legit: google.com (0 digits). Phishing: pay-pal-77.com (2 digits).
    digits = sum(c.isdigit() for c in url)
    features.append(digits)
    
    # 3. Randomness (Entropy)
    # Legit: "amazon" (Low entropy). Phishing: "x7z9q2" (High entropy).
    features.append(get_entropy(url))
    
    return features

# Apply the function to every URL
X = [get_features(url) for url in df['url']]

# --- 4. TRAIN MODEL ---
from xgboost import XGBClassifier

# Replace the RandomForest lines with this:
print("🚀 Training XGBoost Model (The Heavy Artillery)...")
# n_estimators=500 means "try 500 times to improve"
model = XGBClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1)
model.fit(x_train, y_train)

# --- 5. RESULTS ---
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"\n🎉 Model Accuracy: {acc * 100:.2f}%")

# --- 6. SAVE ---
joblib.dump(model, 'phishing_model.pkl')
print("💾 Saved 'phishing_model.pkl' (The Brain)")