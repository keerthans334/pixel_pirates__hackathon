document.addEventListener('DOMContentLoaded', function() {
    const trainButton = document.getElementById('train');
    const realtimeButton = document.getElementById('realtime');
    const speechButton = document.getElementById('speech');
    const clearCacheButton = document.getElementById('clearCache');
    const cacheResult = document.getElementById('cacheResult');

    // Redirect to respective pages
    if (trainButton) {
        trainButton.addEventListener('click', function() {
            window.location.href = 'train.html';
        });
    }

    if (realtimeButton) {
        realtimeButton.addEventListener('click', function() {
            window.location.href = 'realtime.html';
        });
    }

    if (speechButton) {
        speechButton.addEventListener('click', function() {
            window.location.href = 'speech.html';
        });
    }

    // Clear cache functionality
    if (clearCacheButton) {
        clearCacheButton.addEventListener('click', function() {
            fetch('/clear_cache')
                .then(response => response.json())
                .then(data => {
                    if (cacheResult) {
                        cacheResult.textContent = data.message;
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                });
        });
    }

    // Add gesture form functionality (only on train.html)
    const gestureForm = document.getElementById("gestureForm");
    if (gestureForm) {
        gestureForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const gestures = document.getElementById("gestures").value.split(",").map(g => g.trim());
            fetch("/train", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ gestures }),
            })
                .then(response => response.json())
                .then(data => {
                    const trainingResult = document.getElementById("trainingResult");
                    if (trainingResult) {
                        trainingResult.innerText = data.message;
                    }
                })
                .catch(error => {
                    console.error("Error:", error);
                });
        });
    }
});