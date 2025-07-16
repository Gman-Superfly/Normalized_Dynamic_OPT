// Logic to run analysis and display results will go here.

// Utility function to get metric direction indicator
function getMetricDirection(metricName) {
    const lowerIsBetter = [
        'trajectory_smoothness', 'trajectory smoothness',
        'runtime', 'execution time', 'time',
        'error', 'loss', 'variance'
    ];
    
    const higherIsBetter = [
        'local_structure', 'local structure',
        'cell_type_preservation', 'cell type preservation',
        'spatial_coherence', 'spatial coherence', 
        'temporal_ordering', 'temporal ordering',
        'overall_score', 'overall score', 'biological score',
        'accuracy', 'precision', 'recall', 'f1'
    ];
    
    const metric = metricName.toLowerCase();
    
    if (lowerIsBetter.some(term => metric.includes(term))) {
        return '(lower is better)';
    } else if (higherIsBetter.some(term => metric.includes(term))) {
        return '(higher is better)';
    }
    
    return ''; // No indicator if metric type is unknown
}

// Function to add direction indicators to metric displays
function addMetricDirections() {
    // Add to any element with data-metric attribute
    document.querySelectorAll('[data-metric]').forEach(element => {
        const metricName = element.getAttribute('data-metric');
        const direction = getMetricDirection(metricName);
        if (direction && !element.textContent.includes('better')) {
            element.innerHTML += ` <small class="metric-direction">${direction}</small>`;
        }
    });
} 

document.addEventListener('DOMContentLoaded', () => {
    const runGaiaButton = document.getElementById('run-analysis-btn');
    const runWineButton = document.getElementById('run-wine-btn');
    const resultsContainer = document.getElementById('results-container');

    // Only add GAIA button listener if the button exists on this page
    if (runGaiaButton) {
        runGaiaButton.addEventListener('click', () => {
            const selectedSize = document.querySelector('input[name="dataset_size"]:checked').value;
            runAnalysis('/run', { size: selectedSize }, runGaiaButton, 'Run Gaia Analysis');
        });
    }

    // Only add wine button listener if the button exists on this page
    if (runWineButton) {
        runWineButton.addEventListener('click', () => {
            runAnalysis('/run_wine', {}, runWineButton, 'Run Wine Dataset Analysis');
        });
    }

    function runAnalysis(endpoint, body, button, buttonText) {
        // Show a loading state
        button.disabled = true;
        button.textContent = 'Running Analysis...';
        resultsContainer.innerHTML = '<div class="result-block"><p>Processing... this may take a moment.</p></div>';

        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        })
        .then(response => response.json())
        .then(data => {
            resultsContainer.innerHTML = ''; // Clear the "Processing..." message
            if (data.success) {
                displayResults(data);
            } else {
                displayError(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayError({ error: 'An unexpected error occurred. Please check the console.' });
        })
        .finally(() => {
            button.disabled = false;
            button.textContent = buttonText;
        });
    }

    function displayResults(data) {
        let timingsHtml = '<ul>';
        for (const [model, time] of Object.entries(data.timings)) {
            timingsHtml += `<li>${model}: ${time}s</li>`;
        }
        timingsHtml += '</ul>';

        // Determine the title based on the data provided
        let title = 'Analysis Results';
        if (data.sample_count && data.image_path.includes('GAIA')) {
            title = `GAIA Results for ${data.sample_count} Samples`;
        } else if (data.image_path.includes('wine')) {
            title = `Wine Dataset Results (${data.sample_count} samples)`;
        }

        const resultBlock = document.createElement('div');
        resultBlock.className = 'result-block';
        resultBlock.innerHTML = `
            <h2>${title}</h2>
            <div>
                <h3>Execution Times:</h3>
                ${timingsHtml}
            </div>
            <h3>Execution Log:</h3>
            <pre class="terminal-log">${data.logs}</pre>
            <img src="${'static/' + data.image_path.replace(/\\\\/g, '/')}" alt="Analysis Result Plot">
        `;

        resultsContainer.prepend(resultBlock);
    }

    function displayError(data) {
        let errorContent = `<p style="color: #ff4d4d;">Error: ${data.error}</p>`;
        if (data.logs) {
            errorContent += `
                <h3>Execution Log:</h3>
                <pre class="terminal-log">${data.logs}</pre>
            `;
        }
        resultsContainer.innerHTML = `<div class="result-block">${errorContent}</div>`;
    }
}); 