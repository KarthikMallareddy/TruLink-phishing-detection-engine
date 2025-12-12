// 1. Get the URL of the active tab
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    let currentUrl = tabs[0].url;
    document.getElementById("url-display").innerText = currentUrl;

    // 2. Send URL to your Python "Brain" (FastAPI)
    fetch("http://127.0.0.1:8000/check", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ url: currentUrl })
    })
    .then(response => response.json())
    .then(data => {
        let box = document.getElementById("status-box");
        let score = document.getElementById("confidence-score");

        // 3. Update UI based on AI result
        if (data.result === "PHISHING") {
            box.innerText = "⚠️ UNSAFE";
            box.className = "danger"; // Turns RED
            score.innerText = `AI Confidence: ${data.confidence}`;
        } else {
            box.innerText = "✅ SAFE";
            box.className = "safe"; // Turns GREEN
            score.innerText = `AI Confidence: ${data.confidence}`;
        }
    })
    .catch(error => {
        let box = document.getElementById("status-box");
        let score = document.getElementById("confidence-score");
        
        box.innerText = "ERROR";
        box.className = "loading";
        score.innerText = "Is 'app.py' running?";
        console.error("Error:", error);
    });
});