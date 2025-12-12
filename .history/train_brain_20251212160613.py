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

def get_features(url):
    # This function turns text into numbers
    features = []
    url = str(url)
    
    # 1. Length of URL
    features.append(len(url))
    # 2. Count dots (google.com = 1, secure.login.paypal.com = 3)
    features.append(url.count('.'))
    # 3. Has @ symbol?
    features.append(1 if '@' in url else 0)
    # 4. Has IP address?
    features.append(1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0)
    # 5. Count slashes (depth of path)
    features.append(url.count('/'))
    # 6. Has 'https'?
    features.append(1 if 'https' in url else 0)
    
    return features

# Apply the function to every URL
X = [get_features(url) for url in df['url']]

# --- 4. TRAIN MODEL ---
print("🚀 Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

# --- 5. RESULTS ---
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"\n🎉 Model Accuracy: {acc * 100:.2f}%")

# --- 6. SAVE ---
joblib.dump(model, 'phishing_model.pkl')
print("💾 Saved 'phishing_model.pkl' (The Brain)")