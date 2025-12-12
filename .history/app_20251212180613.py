from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import math
import scipy.sparse as sp
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
print("⏳ Loading model files...")
try:
    model = joblib.load('hybrid_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Model loaded!")
except:
    print("❌ ERROR: Could not find .pkl files.")

# --- WHITELIST (Add any trusted site here that gets flagged falsely) ---
WHITELIST = [
    "airtelxstream.in",
    "google.com",
    "youtube.com",
    "facebook.com",
    "amazon.com",
    "wikipedia.org",
    "netflix.com",
    "linkedin.com",
    "microsoft.com",
    "twitter.com",
    "instagram.com"
]

class UrlRequest(BaseModel):
    url: str

# --- FEATURE EXTRACTION ---
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
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('/'))
    features.append(1 if '@' in url else 0)
    features.append(1 if 'https' in url else 0)
    features.append(url.count('-'))
    features.append(url.count('?'))
    features.append(url.count('='))
    
    sketchy = 0
    for tld in ['.xyz', '.top', '.club', '.win', '.info']:
        if url.endswith(tld): sketchy = 1
    features.append(sketchy)
    
    sus_words = ['login', 'secure', 'account', 'update', 'banking']
    hits = 0
    for word in sus_words:
        if word in url: hits += 1
    features.append(hits)
    
    features.append(get_entropy(url))
    return [features]

@app.post("/check")
def check_url(req: UrlRequest):
    url_lower = req.url.lower()

    # --- CHECK 1: INTERNAL PAGES ---
    if url_lower.startswith(("chrome://", "edge://", "about:", "file://")):
        return {"result": "SAFE", "confidence": "100% (System)"}

    # --- CHECK 2: WHITELIST ---
    for domain in WHITELIST:
        if domain in url_lower:
            return {"result": "SAFE", "confidence": "100% (Trusted)"}

    # --- CHECK 3: AI PREDICTION ---
    text_vector = vectorizer.transform([req.url])
    manual_features = get_manual_features(req.url)
    manual_vector = sp.csr_matrix(manual_features)
    final_input = sp.hstack([text_vector, manual_vector])
    
    prediction = model.predict(final_input)[0]
    probs = model.predict_proba(final_input)[0]
    confidence = max(probs) * 100
    
    result_text = "PHISHING" if prediction == 1 else "SAFE"
    
    return {
        "result": result_text,
        "confidence": f"{confidence:.2f}%"
    }