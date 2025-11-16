// Get DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMsg = document.getElementById('errorMsg');

let selectedFile = null;

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag over event
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

// Drag leave event
uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

// Drop event
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// File input change event
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file upload
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    selectedFile = file;
    
    // Preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewSection.style.display = 'block';
        analyzeBtn.style.display = 'block';
        results.style.display = 'none';
        errorMsg.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze button click event
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading state
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    results.style.display = 'none';
    errorMsg.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (err) {
        showError('Failed to connect to server: ' + err.message);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

// Display analysis results
function displayResults(data) {
    document.getElementById('classification').textContent = data.classification;
    document.getElementById('confidence').textContent = data.confidence;
    document.getElementById('ecoScore').textContent = data.eco_score;
    document.getElementById('decompose').textContent = data.decompose_time;
    document.getElementById('tip').textContent = data.tip;
    document.getElementById('wasteType').textContent = 
        data.classification.includes('Biodegradable') ? 'Organic' : 'Recyclable';
    
    results.style.display = 'block';
}

// Show error message
function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
}