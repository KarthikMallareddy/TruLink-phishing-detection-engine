# TruLink AI - Phishing Link Detection Engine

## Project Overview
TruLink AI is a real-time cybersecurity browser extension designed to detect phishing websites and malicious URLs with high precision. Unlike standard blocklists that rely on static databases, TruLink AI employs a hybrid Machine Learning architecture. It combines Natural Language Processing (NLP) with forensic feature engineering to analyze the structure and content of a URL instantly, providing a safety probability score.

The system operates on a Client-Server architecture:
1. Client: A Google Chrome Extension acting as the user interface.
2. Server: A cloud-deployed Python API (FastAPI) hosting the trained XGBoost model.

## Key Features
- **Hybrid AI Analysis:** distinct from simple classifiers, this system merges TF-IDF Vectorization (text pattern recognition) with Manual Feature Engineering (entropy, symbol counts, URL length) to detect complex threats.
- **XGBoost Classifier:** Powered by an eXtreme Gradient Boosting model trained on over 500,000 URLs, functioning as a committee of decision trees for accurate verdicts.
- **Real-Time Scanning:** Delivers instant safety assessments with a specific confidence percentage (e.g., "98.5% Unsafe").
- **Whitelist Logic:** Includes a localized whitelist to prevent false positives on trusted government, educational, and known banking domains.
- **Cloud Architecture:** The backend is deployed on Render, allowing the extension to function globally without local server dependencies.

## Technical Stack

### Frontend (Chrome Extension)
- **HTML5 & CSS3:** User interface and visual status indicators.
- **JavaScript (ES6):** Handles DOM manipulation and asynchronous API calls (fetch).
- **Chrome Extension API:** Utilizes activeTab and scripting permissions.

### Backend (API & Logic)
- **Python 3.x:** Core programming language.
- **FastAPI:** High-performance web framework for API endpoints.
- **Uvicorn:** ASGI server for production deployment.

### Machine Learning (The Brain)
- **XGBoost:** Primary classification model.
- **Scikit-Learn:** Used for TF-IDF vectorization and data preprocessing.
- **Pandas & NumPy:** Data manipulation and matrix operations.
- **SciPy:** Sparse matrix calculations for feature stacking.
- **Joblib:** Model serialization and loading.

## System Architecture
The application follows a strict data processing pipeline:

1. Input: The user activates the extension. The current tab URL is captured.
2. Transmission: The URL is packaged as a JSON payload and sent via HTTPS POST to the cloud API.
3. Preprocessing:
   - The URL is tokenized into character N-Grams (3-5 characters).
   - Forensic features (Entropy, Length, Dot Count) are calculated mathematically.
4. Vectorization: Text data is converted into numerical values using the pre-trained vectorizer mapping.
5. Prediction: The XGBoost model analyzes the combined data vector and outputs a probability score.
6. Response: The verdict (SAFE/PHISHING) and confidence score are returned to the extension.

## Installation and Setup

### Prerequisites
- Google Chrome Browser
- Git
- Python 3.9+ (For local development)

### Option 1: User Installation (Cloud API)
This method uses the live Render server. No local Python setup is required.

1. Download the extension folder or clone this repository.
2. Open Google Chrome and navigate to chrome://extensions.
3. Enable "Developer Mode" in the top-right corner.
4. Click "Load Unpacked".
5. Select the extension folder from the directory.
6. The icon will appear in the browser toolbar.

### Option 2: Developer Installation (Local Backend)
To modify the AI model or API logic, run the server locally.

1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/SafeLink-AI-Detector.git
   cd SafeLink-AI-Detector

2. Install dependencies:
   pip install -r requirements.txt

3. Start the local server:
   uvicorn app:app --reload

4. Update the Extension:
   - Open extension/popup.js.
   - Change the fetch URL to http://127.0.0.1:8000/check.

## File Structure Description
- **app.py:** The main entry point for the FastAPI backend server. Handles routing and validation.
- **train_brain.py:** The script used to train the machine learning model. Running this file generates new .pkl files.
- **requirements.txt:** List of Python dependencies required for cloud deployment.
- **hybrid_model.pkl:** The frozen XGBoost model file containing the decision trees.
- **vectorizer.pkl:** The frozen TF-IDF vectorizer containing the vocabulary mapping.
- **extension/**: Source code for the Chrome Extension.
  - **manifest.json:** Configuration file defining permissions and icons.
  - **popup.html:** The frontend layout.
  - **popup.js:** The logic script connecting the browser to the API.

## Deployment Guide
This project is configured for deployment on Render.

1. Push code to GitHub, ensuring requirements.txt is present.
2. Create a new Web Service on Render.
3. Connect the GitHub repository.
4. Set Build Command: pip install -r requirements.txt
5. Set Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT

## Future Roadmap
- **Visual Analysis:** Implementation of Computer Vision (CNNs) to compare screenshots of suspicious sites against known login pages.
- **Crowdsourcing:** A reporting mechanism for users to flag new phishing domains.
- **Database Caching:** Integration with Redis to cache results for frequently visited URLs, reducing API load.

## Disclaimer
TruLink AI provides a probabilistic assessment based on learned patterns. While highly accurate, no automated detection system is perfect. Users should exercise caution when sharing sensitive information online.
