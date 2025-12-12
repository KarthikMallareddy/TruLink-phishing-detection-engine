# app.py (The Final Hybrid Version)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import math
import scipy.sparse as sp

app = FastAPI()

# 1. LOAD THE BRAINS (Model + Vectorizer)
print("⏳ Loading model files...")
model = joblib.load('hybrid_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("✅ Model loaded!")

class UrlRequest(BaseModel):
    url: str

# 2. FEATURE EXTRACTOR (Must be identical to train_brain.py)
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
    # 1. Vectorize Text
    text_vector = vectorizer.transform([req.url])
    
    # 2. Extract Manual Features
    manual_features = get_manual_features(req.url)
    manual_vector = sp.csr_matrix(manual_features)
    
    # 3. Stack Them
    final_input = sp.hstack([text_vector, manual_vector])
    
    # 4. Predict
    prediction = model.predict(final_input)[0]
    probs = model.predict_proba(final_input)[0]
    confidence = max(probs) * 100
    
    result_text = "PHISHING" if prediction == 1 else "SAFE"
    
    return {
        "result": result_text,
        "confidence": f"{confidence:.2f}%"
    }