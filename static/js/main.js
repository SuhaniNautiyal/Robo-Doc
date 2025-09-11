// =======================
// RoboDoc Emergency Medical Assistant - Phase 2
// =======================

// Global variables
let isRecording = false;
let recognition;
let currentRating = 0;
let userLocation = null;

let currentSession = {
    symptoms: '',
    timestamp: new Date(),
    sessionId: generateSessionId(),
    imageData: null,
    analysis: null,
    imageAnalysis: null
};

// =======================
// Utility Functions
// =======================

// Generate unique session ID
function generateSessionId() {
    return 'guest_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Show/hide loading overlay
function showLoading(show, message = 'Analyzing symptoms...') {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (show) {
        loadingOverlay.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-robot fa-spin"></i>
                <p>${message}</p>
                <div class="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                </div>
            </div>
        `;
        loadingOverlay.style.display = 'flex';
    } else {
        loadingOverlay.style.display = 'none';
    }
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

// =======================
// Speech Recognition
// =======================
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            document.getElementById('voiceStatus').textContent = '🎤 Listening...';
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.getElementById('symptomInput').value += (document.getElementById('symptomInput').value ? ' ' : '') + transcript;
            document.getElementById('voiceStatus').textContent = '✅ Voice input captured';
            setTimeout(() => {
                document.getElementById('voiceStatus').textContent = '';
            }, 3000);
        };

        recognition.onerror = function(event) {
            document.getElementById('voiceStatus').textContent = '❌ Voice recognition error';
            console.error('Speech recognition error:', event.error);
        };

        recognition.onend = function() {
            isRecording = false;
            updateVoiceButton();
        };
    } else {
        console.log('Speech recognition not supported');
        document.getElementById('voiceBtn').style.display = 'none';
    }
}

function toggleVoice() {
    if (!recognition) {
        alert('Voice recognition not supported in your browser');
        return;
    }

    if (isRecording) {
        recognition.stop();
        isRecording = false;
    } else {
        recognition.start();
        isRecording = true;
    }

    updateVoiceButton();
}

function updateVoiceButton() {
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceText = document.getElementById('voiceText');
    if (isRecording) {
        voiceBtn.classList.add('recording');
        voiceText.textContent = 'Stop Recording';
    } else {
        voiceBtn.classList.remove('recording');
        voiceText.textContent = 'Voice Input';
    }
}

// =======================
// Image Upload
// =======================
function handleImageUpload() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
        alert('Image too large. Please select an image under 5MB.');
        return;
    }

    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        currentSession.imageData = e.target.result;
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = `
            <img src="${e.target.result}" alt="Uploaded image">
            <div class="preview-info">
                <p>✅ Image uploaded: ${file.name}</p>
                <button onclick="removeImage()" style="background: #e74c3c; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; margin-top: 5px;">
                    <i class="fas fa-times"></i> Remove
                </button>
            </div>
        `;
        document.querySelector('.analyze-btn').innerHTML = `
            <i class="fas fa-search"></i>
            <span id="analyzeText">Analyze Symptoms + Image</span>
        `;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    currentSession.imageData = null;
    document.getElementById('imagePreview').innerHTML = '';
    document.getElementById('imageInput').value = '';
    document.querySelector('.analyze-btn').innerHTML = `
        <i class="fas fa-search"></i>
        <span id="analyzeText">Analyze Symptoms</span>
    `;
}

// =======================
// Symptom Analysis
// =======================
async function analyzeSymptoms() {
    const symptoms = document.getElementById('symptomInput').value.trim();
    const useAI = document.getElementById('useAI').checked;

    if (!symptoms && !currentSession.imageData) {
        alert('Please describe symptoms or upload an image first.');
        return;
    }

    currentSession.symptoms = symptoms;
    currentSession.timestamp = new Date();

    showLoading(true, useAI ? 'AI analyzing symptoms...' : 'Analyzing symptoms...');
    document.querySelector('.analyze-btn').classList.add('analyzing');
    document.getElementById('analyzeText').textContent = 'Analyzing...';

    try {
        let analysisPromises = [];
        if (symptoms) {
            analysisPromises.push(fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symptoms, sessionId: currentSession.sessionId, useAI })
            }));
        }
        if (currentSession.imageData) {
            analysisPromises.push(fetch('/analyze-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageData: currentSession.imageData, sessionId: currentSession.sessionId })
            }));
        }

        const responses = await Promise.all(analysisPromises);
        const results = await Promise.all(responses.map(r => r.json()));

        let mainAnalysis = null, imageAnalysis = null;
        if (symptoms && results[0] && results[0].success) mainAnalysis = results[0].analysis;
        if (currentSession.imageData && results.length > 1 && results[1].success) imageAnalysis = results[1].analysis;

        if (mainAnalysis || imageAnalysis) displayResults(mainAnalysis, imageAnalysis, useAI);
        else showError('Analysis failed. Please try again.');
    } catch (error) {
        console.error('Error analyzing symptoms:', error);
        showError('Connection error. Please check your internet connection.');
    }

    document.querySelector('.analyze-btn').classList.remove('analyzing');
    document.getElementById('analyzeText').textContent = 'Analyze Symptoms';
    showLoading(false);
}

function displayResults(analysis, imageAnalysis = null, aiUsed = false) {
    const resultsSection = document.getElementById('resultsSection');
    const riskLevel = document.getElementById('riskLevel');
    const firstAid = document.getElementById('firstAid');
    const recommendations = document.getElementById('recommendations');
    const aiInsights = document.getElementById('aiInsights');
    const imageResults = document.getElementById('imageResults');

    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    if (analysis) {
        const risk = analysis.riskLevel || 'unknown';
        riskLevel.innerHTML = `
            <h3><i class="fas fa-exclamation-triangle"></i> Risk Assessment</h3>
            <div class="risk-${risk.toLowerCase()}">
                <h4>${risk.toUpperCase()} PRIORITY</h4>
                <p>${analysis.riskDescription}</p>
                ${aiUsed ? '<span class="ai-badge">🤖 AI-Enhanced Analysis</span>' : ''}
            </div>
        `;
        firstAid.innerHTML = `
            <h4><i class="fas fa-first-aid"></i> Immediate Actions</h4>
            <ol>${analysis.firstAidSteps.map(step => `<li>${step}</li>`).join('')}</ol>
        `;
        recommendations.innerHTML = `
            <h4><i class="fas fa-stethoscope"></i> Medical Recommendations</h4>
            <div class="recommendation-content">${analysis.recommendations.map(rec => `<p>• ${rec}</p>`).join('')}</div>
        `;
        if (analysis.aiInsights && aiUsed) {
            aiInsights.style.display = 'block';
            aiInsights.innerHTML = `
                <h4><i class="fas fa-robot"></i> AI Insights</h4>
                <div><strong>Predicted Condition:</strong> ${analysis.aiInsights.predicted_condition}</div>
                <div><strong>Specialist Recommended:</strong> ${analysis.aiInsights.specialist}</div>
                <div class="confidence-meter">
                    <span>Confidence:</span>
                    <div class="confidence-bar"><div class="confidence-fill" style="width:${(analysis.aiInsights.confidence * 100).toFixed(0)}%"></div></div>
                    <span>${(analysis.aiInsights.confidence * 100).toFixed(0)}%</span>
                </div>
                ${analysis.aiInsights.key_symptoms.length > 0 ? `<div><strong>Key Symptoms:</strong> ${analysis.aiInsights.key_symptoms.join(', ')}</div>` : ''}
            `;
        }
    }

    if (imageAnalysis) {
        imageResults.style.display = 'block';
        imageResults.innerHTML = `
            <h4><i class="fas fa-camera"></i> Image Analysis Results</h4>
            <div><strong>Detected Condition:</strong> ${imageAnalysis.detected_condition}
            <span class="severity-indicator severity-${imageAnalysis.severity}">${imageAnalysis.severity.toUpperCase()}</span></div>
            <div class="confidence-meter">
                <span>Analysis Confidence:</span>
                <div class="confidence-bar"><div class="confidence-fill" style="width:${(imageAnalysis.confidence * 100).toFixed(0)}%"></div></div>
                <span>${(imageAnalysis.confidence * 100).toFixed(0)}%</span>
            </div>
            <div><h5>Image-Based Recommendations:</h5>${imageAnalysis.first_aid.map(step => `<p>• ${step}</p>`).join('')}</div>
        `;
    }

    currentSession.analysis = analysis;
    currentSession.imageAnalysis = imageAnalysis;
}

// =======================
// Hospital Finder
// =======================
async function findNearbyHelp() {
    showLoading(true, 'Finding nearby medical facilities...');
    try {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(async function(position) {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                userLocation = { lat, lng };
                const response = await fetch(`/hospitals/nearby?lat=${lat}&lng=${lng}&radius=5000`);
                const data = await response.json();
                if (data.success) displayEnhancedNearbyHelp(data.hospitals, data.center);
                else displayEnhancedNearbyHelp([], null);
                showLoading(false);
            }, function() {
                showLoading(false);
                promptForLocation();
            });
        } else {
            showLoading(false);
            alert('Geolocation not supported.');
        }
    } catch (error) {
        showLoading(false);
        console.error('Error finding nearby help:', error);
        alert('Error finding nearby medical facilities.');
    }
}

function displayEnhancedNearbyHelp(facilities, center) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 800px;">
            <span class="close" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h2><i class="fas fa-hospital"></i> Nearby Medical Help</h2>
            ${center ? `<p><i class="fas fa-map-marker-alt"></i> Near ${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}</p>` : ''}
            <div class="facilities-list" style="max-height:400px;overflow-y:auto;">
                ${facilities.length > 0 ? facilities.map(f => `
                    <div class="facility-item" style="padding:15px;margin:10px 0;border:1px solid #ddd;border-radius:10px;background:white;">
                        <h4>${f.name}</h4>
                        <p><i class="fas fa-map-marker-alt"></i> ${f.address || 'Address not available'}</p>
                        <p><i class="fas fa-route"></i> ${f.distance ? f.distance.toFixed(1)+' km' : 'Unknown distance'} away</p>
                        ${f.rating ? `<p><i class="fas fa-star"></i> ${f.rating}/5</p>` : ''}
                        ${f.emergency_services ? '<span style="color:red;font-weight:bold;">🚨 Emergency Services Available</span>' : ''}
                        <div style="margin-top:10px;">
                            ${f.phone ? `<button onclick="window.location.href='tel:${f.phone}'" style="background:#4a90e2;color:white;padding:5px 10px;border:none;border-radius:5px;">Call</button>` : ''}
                            ${f.coordinates ? `<button onclick="getDirections(${f.coordinates.lat},${f.coordinates.lng})" style="background:#00b894;color:white;padding:5px 10px;border:none;border-radius:5px;">Directions</button>` : ''}
                        </div>
                    </div>
                `).join('') : '<p>No hospitals found nearby</p>'}
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function getDirections(lat, lng) {
    const url = userLocation ? 
        `https://www.google.com/maps/dir/${userLocation.lat},${userLocation.lng}/${lat},${lng}` :
        `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
    window.open(url, '_blank');
}

// =======================
// Feedback
// =======================
function initializeFeedback() {
    const stars = document.querySelectorAll('.star');
    stars.forEach((star, index) => {
        star.addEventListener('click', () => { currentRating = index + 1; updateStarRating(currentRating); });
        star.addEventListener('mouseover', () => updateStarRating(index + 1, true));
    });
    document.querySelector('.star-rating').addEventListener('mouseleave', () => updateStarRating(currentRating));
}

function updateStarRating(rating) {
    const stars = document.querySelectorAll('.star');
    stars.forEach((star, index) => {
        if (index < rating) { star.classList.add('active'); star.textContent = '⭐'; }
        else { star.classList.remove('active'); star.textContent = '☆'; }
    });
}

function showFeedbackModal() {
    if (!currentSession.analysis) {
        alert('Please analyze symptoms first before providing feedback.');
        return;
    }
    const modal = document.getElementById('feedbackModal');
    modal.style.display = 'flex';
    currentRating = 0;
    updateStarRating(0);
    document.getElementById('feedbackText').value = '';
    document.getElementById('analysisHelpful').checked = true;
}

function closeFeedbackModal() {
    document.getElementById('feedbackModal').style.display = 'none';
}

async function submitFeedback() {
    if (currentRating === 0) {
        alert('Please provide a star rating.');
        return;
    }
    const feedbackText = document.getElementById('feedbackText').value;
    const analysisHelpful = document.getElementById('analysisHelpful').checked;
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionId: currentSession.sessionId, rating: currentRating, feedback: feedbackText, analysisHelpful })
        });
        const data = await response.json();
        if (data.success) {
            alert('Thank you for your feedback!');
            closeFeedbackModal();
        } else alert('Failed to submit feedback.');
    } catch (error) {
        console.error('Error submitting feedback:', error);
        alert('Network error.');
    }
}

// =======================
// Init on Load
// =======================
document.addEventListener('DOMContentLoaded', function() {
    initializeSpeechRecognition();
    initializeFeedback();
    if (navigator.geolocation) console.log('Geolocation available');
});
