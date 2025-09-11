# RoboDoc Phase 2 - Flask Backend
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import datetime
import os

# Import AI modules
from ml_analyzer import ml_analyzer
from language_handler import language_handler

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'robodoc_emergency_2024'  # Change in production
CORS(app)

def snake_to_camel(s):
    parts = s.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms with ML + language support"""
    try:
        data = request.get_json(force=True, silent=True)
        print("DEBUG: Received data from frontend:", data)

        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid request format. Expected JSON object.'}), 400

        symptoms = data.get('symptoms', '')
        language = data.get('language', 'en')

        translated_symptoms = language_handler.translate_symptoms(symptoms, language)
        print("DEBUG: Translated symptoms:", translated_symptoms)

        result = ml_analyzer.analyze(translated_symptoms)

        # Convert snake_case keys to camelCase for frontend
        result_camel = {snake_to_camel(k): v for k, v in result.items()}

        return jsonify({'success': True, 'analysis': result_camel})

    except Exception as e:
        print(f"Error in analyze_symptoms: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/hospitals/nearby')
def nearby_hospitals():
    """Mock nearby hospital data (Phase 2 placeholder)"""
    lat = request.args.get('lat', type=float, default=28.61)  # Default: Delhi
    lng = request.args.get('lng', type=float, default=77.21)

    hospitals = [
        {
            'name': 'AIIMS Delhi',
            'address': 'Sri Aurobindo Marg, Ansari Nagar, New Delhi',
            'phone': '011-26588500',
            'distance': 2.5,
            'emergency_services': True,
            'rating': 4.6,
        },
        {
            'name': 'Fortis Hospital',
            'address': 'Sector 62, Noida',
            'phone': '0120-3984444',
            'distance': 4.8,
            'emergency_services': True,
            'rating': 4.3,
        }
    ]

    return jsonify({'success': True, 'hospitals': hospitals, 'center': {'lat': lat, 'lng': lng}})

@app.route('/ai-status')
def ai_status():
    """Check if AI components are ready"""
    return jsonify({
        'ml_analyzer_ready': ml_analyzer.is_trained,
        'language_support': True,
        'supported_languages': list(language_handler.supported_languages.keys()),
        'version': '2.0',
        'features': [
            'ML-based symptom analysis',
            'Multi-language support',
            'Mock hospital finder',
            'Feedback system'
        ]
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Store user feedback"""
    try:
        data = request.get_json()
        feedback_data = {
            'session_id': data.get('sessionId', ''),
            'rating': data.get('rating', 0),
            'feedback': data.get('feedback', ''),
            'analysis_helpful': data.get('analysisHelpful', True),
            'timestamp': datetime.datetime.now().isoformat(),
            'user_agent': request.headers.get('User-Agent'),
        }

        if 'feedbacks' not in session:
            session['feedbacks'] = []
        session['feedbacks'].append(feedback_data)

        print(f"📝 Feedback received: {feedback_data['rating']}/5")

        return jsonify({'success': True, 'message': 'Feedback stored successfully'})

    except Exception as e:
        app.logger.error(f"Error storing feedback: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to store feedback'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')

    print("🤖 Starting RoboDoc Phase 2 Backend...")
    print("📱 Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
