document.addEventListener('DOMContentLoaded', function() {
    
    const inputField = document.getElementById("manual-input");
    const checkBtn = document.getElementById("check-btn");

    // Reusable function to check any URL
    function checkUrl(urlToCheck) {
        document.getElementById("status-box").innerText = "Scanning...";
        document.getElementById("status-box").className = "loading";
        document.getElementById("confidence-score").innerText = "";

        fetch("http://127.0.0.1:8000/check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: urlToCheck })
        })
        .then(res => res.json())
        .then(data => {
            let box = document.getElementById("status-box");
            let score = document.getElementById("confidence-score");

            if (data.result === "PHISHING") {
                box.innerText = "⚠️ UNSAFE";
                box.className = "danger";
            } else {
                box.innerText = "✅ SAFE";
                box.className = "safe";
            }
            score.innerText = `Confidence: ${data.confidence}`;
        })
        .catch(err => {
            document.getElementById("status-box").innerText = "ERROR";
            document.getElementById("status-box").className = "loading";
        });
    }

    // 1. Auto-fill and check current tab when opened
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        let currentUrl = tabs[0].url;
        inputField.value = currentUrl;
        checkUrl(currentUrl);
    });

    // 2. Check manually when button is clicked
    checkBtn.addEventListener("click", function() {
        checkUrl(inputField.value);
    });
});