document.addEventListener('DOMContentLoaded', function () {
    // Back button functionality
    const backButton = document.getElementById('back');
    if (backButton) {
        backButton.addEventListener('click', function () {
            window.location.href = '/'; // Redirect to the home page
        });
    }

    // Train Model button functionality
    const trainButton = document.getElementById('train');
    if (trainButton) {
        trainButton.addEventListener('click', function () {
            window.location.href = '/train.html'; // Redirect to the train page
        });
    }

    // Realtime Recognition button functionality
    const realtimeButton = document.getElementById('realtime');
    if (realtimeButton) {
        realtimeButton.addEventListener('click', function () {
            fetch('/realtime', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    alert(data.message); // Notify the user that recognition has started
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to start realtime recognition.');
                });
        });
    }

    // Speech Recognition button functionality
    const speechButton = document.getElementById('speech');
    if (speechButton) {
        speechButton.addEventListener('click', function () {
            fetch('/speech', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    alert(data.message); // Notify the user that speech recognition has started
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to start speech recognition.');
                });
        });
    }

    // Clear Cache button functionality
    const clearCacheButton = document.getElementById('clearCache');
    if (clearCacheButton) {
        clearCacheButton.addEventListener('click', function () {
            fetch('/clear_cache', {
                method: 'POST'
            })
                .then(response => response.json())
                .then(data => {
                    alert(data.message); // Notify the user that cache has been cleared
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to clear cache.');
                });
        });
    }

    // Confidence threshold controls
    const confidenceThresholdElement = document.getElementById('confidenceThreshold');
    const decreaseConfidenceButton = document.getElementById('decreaseConfidence');
    const increaseConfidenceButton = document.getElementById('increaseConfidence');
    let confidenceThreshold = 0.75; // Default confidence threshold

    if (decreaseConfidenceButton && increaseConfidenceButton && confidenceThresholdElement) {
        decreaseConfidenceButton.addEventListener('click', function () {
            confidenceThreshold = Math.max(0.5, confidenceThreshold - 0.05); // Minimum threshold is 0.5
            confidenceThresholdElement.textContent = `Confidence Threshold: ${Math.round(confidenceThreshold * 100)}%`;
        });

        increaseConfidenceButton.addEventListener('click', function () {
            confidenceThreshold = Math.min(0.95, confidenceThreshold + 0.05); // Maximum threshold is 0.95
            confidenceThresholdElement.textContent = `Confidence Threshold: ${Math.round(confidenceThreshold * 100)}%`;
        });
    }

    // Verbose Mode button functionality
    const verboseModeButton = document.getElementById('verboseMode');
    if (verboseModeButton) {
        verboseModeButton.addEventListener('click', function () {
            alert('Verbose mode toggled. Check the console for detailed logs.');
            // You can add additional functionality here to enable verbose logging
        });
    }

    // Display History button functionality
    const historyButton = document.getElementById('history');
    if (historyButton) {
        historyButton.addEventListener('click', function () {
            alert('Displaying gesture history.');
            // You can add functionality here to fetch and display gesture history
        });
    }
});
