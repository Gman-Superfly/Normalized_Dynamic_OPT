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

    // Kernel selection helpers (optional per page)
    function getKernelType() {
        const selectedKernel = document.querySelector('input[name="kernel_type"]:checked');
        return selectedKernel ? selectedKernel.value : 'exponential';
    }

    function getKernelParam() {
        const paramEl = document.getElementById('kernel-param');
        if (!paramEl) return null;
        const parsed = parseFloat(paramEl.value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function getKernelConfig() {
        const kernelType = getKernelType();
        const kernelParam = getKernelParam();

        const cfg = { kernel_type: kernelType };
        if (kernelType === 'generalized' && kernelParam !== null) cfg.kernel_p = kernelParam;
        if (kernelType === 'student_t' && kernelParam !== null) cfg.kernel_nu = kernelParam;
        if (kernelType === 'rational_quadratic' && kernelParam !== null) cfg.kernel_alpha = kernelParam;

        const betaEl = document.getElementById('kernel-beta');
        if (betaEl) {
            const betaParsed = parseFloat(betaEl.value);
            if (Number.isFinite(betaParsed)) cfg.kernel_beta = betaParsed;
        }
        const autoEl = document.getElementById('learn-kernel-beta');
        if (autoEl) cfg.learn_kernel_beta = Boolean(autoEl.checked);

        return cfg;
    }

    // SmartK (K adaptation) helpers (optional per page)
    function getKAdaptationStrategy() {
        const selected = document.querySelector('input[name="k_adaptation_strategy"]:checked');
        return selected ? selected.value : 'off';
    }

    function getKBase() {
        const el = document.getElementById('k-base');
        if (!el) return null;
        const parsed = parseInt(el.value, 10);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function getSmartKConfig() {
        const strategy = getKAdaptationStrategy();
        const cfg = { k_adaptation_strategy: strategy };
        if (strategy === 'fixed') {
            const kBase = getKBase();
            if (kBase !== null) cfg.k_base = kBase;
        }
        return cfg;
    }

    function updateSmartKUI(strategy) {
        const container = document.getElementById('k-base-container');
        if (!container) return;
        container.style.display = (strategy === 'fixed') ? 'flex' : 'none';
    }

    // Smart sampling helpers (optional per page)
    function getSamplingMethod() {
        const selected = document.querySelector('input[name="sampling_method"]:checked');
        return selected ? selected.value : 'off';
    }

    function getSamplingTargetSize() {
        const el = document.getElementById('sampling-target-size');
        if (!el) return null;
        const parsed = parseInt(el.value, 10);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function getSamplingSpatialWeight() {
        const el = document.getElementById('sampling-spatial-weight');
        if (!el) return null;
        const parsed = parseFloat(el.value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function getSmartSamplingConfig() {
        const method = getSamplingMethod();
        const cfg = { sampling_method: method };

        if (method !== 'off') {
            const target = getSamplingTargetSize();
            if (target !== null) cfg.target_size = target;

            if (method === 'hybrid') {
                const w = getSamplingSpatialWeight();
                if (w !== null) cfg.spatial_weight = w;
            }
        }

        return cfg;
    }

    function updateSmartSamplingUI(method) {
        const targetContainer = document.getElementById('sampling-target-container');
        const hybridContainer = document.getElementById('sampling-hybrid-container');

        if (targetContainer) targetContainer.style.display = (method === 'off') ? 'none' : 'flex';
        if (hybridContainer) hybridContainer.style.display = (method === 'hybrid') ? 'flex' : 'none';
    }

    function updateKernelUI(kernelType) {
        const descEl = document.getElementById('kernel-description');
        const paramContainer = document.getElementById('kernel-param-container');
        const paramLabel = document.getElementById('kernel-param-label');
        const paramEl = document.getElementById('kernel-param');

        if (paramContainer && paramLabel && paramEl) {
            const lastKernelType = paramEl.dataset.lastKernelType;
            if (kernelType === 'generalized') {
                paramContainer.style.display = 'flex';
                paramLabel.textContent = 'p';
                if (lastKernelType !== kernelType) paramEl.value = '1.5';
                paramEl.dataset.lastKernelType = kernelType;
            } else if (kernelType === 'student_t') {
                paramContainer.style.display = 'flex';
                paramLabel.textContent = 'ν';
                if (lastKernelType !== kernelType) paramEl.value = '1.0';
                paramEl.dataset.lastKernelType = kernelType;
            } else if (kernelType === 'rational_quadratic') {
                paramContainer.style.display = 'flex';
                paramLabel.textContent = 'α';
                if (lastKernelType !== kernelType) paramEl.value = '1.0';
                paramEl.dataset.lastKernelType = kernelType;
            } else {
                paramContainer.style.display = 'none';
            }
        }

        if (!descEl) return;

        if (kernelType === 'gaussian') {
            descEl.innerHTML = '<strong>Gaussian:</strong> K = exp(-d² / (2σ²)) - Squared distance decay, standard RBF kernel formula';
        } else if (kernelType === 'generalized') {
            descEl.innerHTML = '<strong>Generalized:</strong> K = exp(-(d^p) / (2σ²)) - Generalized exponential family (tune p)';
        } else if (kernelType === 'student_t') {
            descEl.innerHTML = '<strong>Student-t:</strong> K = (1 + d²/(νσ²))^(-(ν+1)/2) - Heavy-tailed kernel (tune ν)';
        } else if (kernelType === 'rational_quadratic') {
            descEl.innerHTML = '<strong>Rational quadratic:</strong> K = (1 + d²/(2ασ²))^(-α) - Gaussian scale mixture (tune α)';
        } else {
            descEl.innerHTML = '<strong>Exponential:</strong> K = exp(-d / (2σ²)) - Linear distance decay, empirically effective on biological data';
        }
    }

    // Kernel type selection event listeners
    document.querySelectorAll('input[name="kernel_type"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateKernelUI(e.target.value);
        });
    });

    // Initialize UI based on default selection
    updateKernelUI(getKernelType());

    // Only add GAIA button listener if the button exists on this page
    if (runGaiaButton) {
        runGaiaButton.addEventListener('click', () => {
            const selectedSize = document.querySelector('input[name="dataset_size"]:checked').value;
            const kernelCfg = getKernelConfig();
            const smartKCfg = getSmartKConfig();
            const smartSamplingCfg = getSmartSamplingConfig();
            runAnalysis('/run', { size: selectedSize, ...kernelCfg, ...smartKCfg, ...smartSamplingCfg }, runGaiaButton, 'Run Gaia Analysis');
        });
    }

    // Only add wine button listener if the button exists on this page
    if (runWineButton) {
        runWineButton.addEventListener('click', () => {
            const kernelCfg = getKernelConfig();
            const smartKCfg = getSmartKConfig();
            const smartSamplingCfg = getSmartSamplingConfig();
            runAnalysis('/run_wine', { ...kernelCfg, ...smartKCfg, ...smartSamplingCfg }, runWineButton, 'Run Wine Dataset Analysis');
        });
    }

    // SmartK strategy listeners (if present)
    document.querySelectorAll('input[name="k_adaptation_strategy"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateSmartKUI(e.target.value);
        });
    });

    // Initialize SmartK UI
    updateSmartKUI(getKAdaptationStrategy());

    // Smart sampling listeners (if present)
    document.querySelectorAll('input[name="sampling_method"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateSmartSamplingUI(e.target.value);
        });
    });

    // Initialize smart sampling UI
    updateSmartSamplingUI(getSamplingMethod());

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