from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import math
import scipy.sparse as sp
# NEW: Import CORS middleware to allow Chrome to talk to this server
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

# --- 1. SETUP CORS (The Fix for "Access Blocked") ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL connections (Chrome, external sites, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],  # Allows all header types
)

# --- 2. LOAD THE BRAIN ---
print("⏳ Loading model files...")
try:
    # Load the trained model and the text vectorizer
    model = joblib.load('hybrid_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Did you run 'train_brain.py' to generate the .pkl files?")

# Define the data shape we expect from Chrome
class UrlRequest(BaseModel):
    url: str

# --- 3. FEATURE EXTRACTION (Must match train_brain.py EXACTLY) ---
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
    
    # Structural features
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('/'))
    features.append(1 if '@' in url else 0)
    features.append(1 if 'https' in url else 0)
    features.append(url.count('-'))
    features.append(url.count('?'))
    features.append(url.count('='))
    
    # Sketchy TLD check
    sketchy = 0
    for tld in ['.xyz', '.top', '.club', '.win', '.info']:
        if url.endswith(tld): sketchy = 1
    features.append(sketchy)
    
    # Suspicious word check
    sus_words = ['login', 'secure', 'account', 'update', 'banking']
    hits = 0
    for word in sus_words:
        if word in url: hits += 1
    features.append(hits)
    
    # Entropy check
    features.append(get_entropy(url))
    
    return [features]

# --- 4. THE API ENDPOINT ---
@app.post("/check")
def check_url(req: UrlRequest):
    # Step A: Convert URL text to numbers (Vectorization)
    text_vector = vectorizer.transform([req.url])
    
    # Step B: Extract manual features (Forensics)
    manual_features = get_manual_features(req.url)
    manual_vector = sp.csr_matrix(manual_features)
    
    # Step C: Combine them (Stacking)
    final_input = sp.hstack([text_vector, manual_vector])
    
    # Step D: Predict
    prediction = model.predict(final_input)[0]
    probs = model.predict_proba(final_input)[0]
    confidence = max(probs) * 100
    
    # Step E: Format Result
    result_text = "PHISHING" if prediction == 1 else "SAFE"
    
    return {
        "result": result_text,
        "confidence": f"{confidence:.2f}%"
    }