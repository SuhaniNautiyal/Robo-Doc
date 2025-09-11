# RoboDoc Advanced AI Engine - Phase 2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import spacy
from textblob import TextBlob
import re
import json
import os
from typing import Dict, List, Tuple, Any
import pickle
from datetime import datetime
import cv2
from PIL import Image
import base64
import io

class AdvancedSymptomAnalyzer:
    """Advanced AI-powered symptom analyzer using multiple ML models"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.nb_classifier = MultinomialNB()
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.nlp = None
        self.is_trained = False
        
        # Load spacy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features may not work.")
        
        # Medical condition database (expanded)
        self.medical_conditions = {
            'cardiac': {
                'symptoms': [
                    'chest pain', 'shortness of breath', 'heart palpitations',
                    'chest tightness', 'pain radiating to arm', 'sweating',
                    'nausea with chest pain', 'irregular heartbeat'
                ],
                'severity': 'critical',
                'first_aid': [
                    'Call 911 immediately',
                    'Help patient sit upright',
                    'Loosen tight clothing',
                    'Give aspirin if not allergic (and conscious)',
                    'Prepare for CPR if needed',
                    'Monitor vital signs'
                ],
                'specialist': 'Cardiologist'
            },
            
            'respiratory': {
                'symptoms': [
                    'difficulty breathing', 'wheezing', 'coughing blood',
                    'severe cough', 'blue lips', 'chest pain when breathing',
                    'rapid breathing', 'cannot speak full sentences'
                ],
                'severity': 'urgent',
                'first_aid': [
                    'Keep patient upright',
                    'Ensure airway is clear',
                    'Use prescribed inhaler if available',
                    'Call emergency services if severe',
                    'Monitor breathing rate',
                    'Stay calm and reassure patient'
                ],
                'specialist': 'Pulmonologist'
            },
            
            'neurological': {
                'symptoms': [
                    'severe headache', 'confusion', 'dizziness', 'seizure',
                    'loss of consciousness', 'slurred speech', 'weakness',
                    'numbness', 'vision problems', 'memory loss'
                ],
                'severity': 'critical',
                'first_aid': [
                    'Ensure patient safety',
                    'Place in recovery position if unconscious',
                    'Do not restrain during seizures',
                    'Time the seizure duration',
                    'Call 911 for severe symptoms',
                    'Stay with patient'
                ],
                'specialist': 'Neurologist'
            },
            
            'gastrointestinal': {
                'symptoms': [
                    'severe abdominal pain', 'vomiting blood', 'black stool',
                    'persistent vomiting', 'dehydration', 'stomach cramps',
                    'bloating', 'loss of appetite', 'acid reflux'
                ],
                'severity': 'moderate',
                'first_aid': [
                    'Keep patient hydrated',
                    'Avoid solid foods temporarily',
                    'Monitor for signs of dehydration',
                    'Seek medical attention if severe',
                    'Rest in comfortable position',
                    'Avoid medications without consultation'
                ],
                'specialist': 'Gastroenterologist'
            },
            
            'infectious': {
                'symptoms': [
                    'high fever', 'chills', 'body aches', 'fatigue',
                    'sore throat', 'cough', 'runny nose', 'swollen glands',
                    'skin rash', 'joint pain'
                ],
                'severity': 'moderate',
                'first_aid': [
                    'Get plenty of rest',
                    'Stay hydrated',
                    'Monitor temperature',
                    'Isolate if contagious',
                    'Take fever reducers as appropriate',
                    'Consult healthcare provider'
                ],
                'specialist': 'Internal Medicine'
            }
        }
        
        # Generate training data
        self._generate_training_data()
        self._train_models()
    
    def _generate_training_data(self):
        """Generate training data from medical conditions database"""
        training_texts = []
        training_labels = []
        
        # Create synthetic training data
        for condition, data in self.medical_conditions.items():
            for symptom in data['symptoms']:
                # Create variations of symptom descriptions
                variations = [
                    f"I have {symptom}",
                    f"Experiencing {symptom}",
                    f"Patient has {symptom}",
                    f"Suffering from {symptom}",
                    f"{symptom} for the past few hours",
                    f"Severe {symptom}",
                    f"Mild {symptom}",
                    f"{symptom} with other issues"
                ]
                
                for variation in variations:
                    training_texts.append(variation)
                    training_labels.append(condition)
        
        self.training_data = pd.DataFrame({
            'symptoms': training_texts,
            'condition': training_labels
        })
    
    def _train_models(self):
        """Train ML models with generated data"""
        try:
            X = self.training_data['symptoms']
            y = self.training_data['condition']
            
            # Vectorize text data
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42
            )
            
            # Train classifiers
            self.nb_classifier.fit(X_train, y_train)
            self.rf_classifier.fit(X_train, y_train)
            
            # Evaluate models
            nb_score = accuracy_score(y_test, self.nb_classifier.predict(X_test))
            rf_score = accuracy_score(y_test, self.rf_classifier.predict(X_test))
            
            print(f"✅ AI Models Trained Successfully!")
            print(f"📊 Naive Bayes Accuracy: {nb_score:.2%}")
            print(f"📊 Random Forest Accuracy: {rf_score:.2%}")
            
            self.is_trained = True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            self.is_trained = False
    
    def analyze_with_ai(self, symptoms_text: str) -> Dict[str, Any]:
        """Advanced AI-powered symptom analysis"""
        if not self.is_trained:
            return self._fallback_analysis(symptoms_text)
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(symptoms_text)
            
            # Vectorize input
            text_vectorized = self.vectorizer.transform([cleaned_text])
            
            # Get predictions from both models
            nb_prediction = self.nb_classifier.predict(text_vectorized)[0]
            nb_probabilities = self.nb_classifier.predict_proba(text_vectorized)[0]
            
            rf_prediction = self.rf_classifier.predict(text_vectorized)[0]
            rf_probabilities = self.rf_classifier.predict_proba(text_vectorized)[0]
            
            # Combine predictions (ensemble approach)
            final_prediction = nb_prediction  # Use NB as primary
            max_probability = max(nb_probabilities)
            
            # Get condition data
            condition_data = self.medical_conditions.get(final_prediction, {})
            
            # Perform sentiment analysis for urgency
            urgency_score = self._analyze_urgency(symptoms_text)
            
            # Extract key symptoms using NLP
            key_symptoms = self._extract_key_symptoms(symptoms_text)
            
            # Generate comprehensive analysis
            analysis = {
                'predicted_condition': final_prediction,
                'confidence': float(max_probability),
                'urgency_score': urgency_score,
                'key_symptoms': key_symptoms,
                'severity': condition_data.get('severity', 'moderate'),
                'first_aid_steps': condition_data.get('first_aid', []),
                'recommended_specialist': condition_data.get('specialist', 'General Physician'),
                'model_insights': {
                    'naive_bayes_prediction': nb_prediction,
                    'random_forest_prediction': rf_prediction,
                    'nb_confidence': float(max(nb_probabilities)),
                    'rf_confidence': float(max(rf_probabilities))
                },
                'timestamp': datetime.now().isoformat(),
                'analysis_version': '2.0'
            }
            
            return self._format_analysis_response(analysis)
            
        except Exception as e:
            print(f"AI Analysis error: {e}")
            return self._fallback_analysis(symptoms_text)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess symptom text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _analyze_urgency(self, text: str) -> float:
        """Analyze urgency based on language sentiment and keywords"""
        urgency_keywords = {
            'critical': ['severe', 'intense', 'unbearable', 'excruciating', 'emergency'],
            'urgent': ['sudden', 'sharp', 'worsening', 'getting worse', 'can\'t'],
            'moderate': ['mild', 'occasional', 'sometimes', 'slight', 'minor']
        }
        
        urgency_score = 0.5  # Base score
        
        text_lower = text.lower()
        
        # Check for urgency keywords
        for level, keywords in urgency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'critical':
                        urgency_score += 0.3
                    elif level == 'urgent':
                        urgency_score += 0.2
                    elif level == 'moderate':
                        urgency_score -= 0.1
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        
        # Negative sentiment often indicates more serious symptoms
        if sentiment_polarity < -0.2:
            urgency_score += 0.2
        
        return min(1.0, max(0.0, urgency_score))
    
    def _extract_key_symptoms(self, text: str) -> List[str]:
        """Extract key medical symptoms using NLP"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        symptoms = []
        
        # Extract noun phrases that might be symptoms
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep short phrases
                symptoms.append(chunk.text.lower())
        
        # Extract important adjective-noun combinations
        for token in doc:
            if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                symptom = f"{token.text} {token.head.text}".lower()
                if symptom not in symptoms:
                    symptoms.append(symptom)
        
        return symptoms[:10]  # Limit to top 10
    
    def _format_analysis_response(self, analysis: Dict) -> Dict[str, Any]:
        """Format analysis response for frontend"""
        severity = analysis['severity']
        urgency = analysis['urgency_score']
        
        # Determine risk level
        if severity == 'critical' or urgency > 0.8:
            risk_level = 'HIGH'
            risk_description = f"Critical condition detected: {analysis['predicted_condition']}. Immediate medical attention required."
        elif severity == 'urgent' or urgency > 0.6:
            risk_level = 'MEDIUM' 
            risk_description = f"Urgent medical condition: {analysis['predicted_condition']}. Seek medical care within 24 hours."
        else:
            risk_level = 'LOW'
            risk_description = f"Moderate condition: {analysis['predicted_condition']}. Monitor symptoms and consult healthcare provider."
        
        # Generate recommendations
        recommendations = [
            f"Consult a {analysis['recommended_specialist']} for specialized care",
            f"Condition confidence: {analysis['confidence']:.1%}",
            "This is an AI analysis - always consult medical professionals for definitive diagnosis"
        ]
        
        if analysis['key_symptoms']:
            recommendations.append(f"Key symptoms identified: {', '.join(analysis['key_symptoms'][:3])}")
        
        return {
            'riskLevel': risk_level,
            'riskDescription': risk_description,
            'firstAidSteps': analysis['first_aid_steps'],
            'recommendations': recommendations,
            'aiInsights': {
                'predicted_condition': analysis['predicted_condition'],
                'confidence': analysis['confidence'],
                'urgency_score': urgency,
                'specialist': analysis['recommended_specialist'],
                'key_symptoms': analysis['key_symptoms']
            },
            'timestamp': analysis['timestamp'],
            'version': '2.0-AI'
        }
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis if AI models fail"""
        return {
            'riskLevel': 'MEDIUM',
            'riskDescription': 'Basic analysis performed. Consult healthcare provider for proper evaluation.',
            'firstAidSteps': [
                'Monitor symptoms closely',
                'Seek medical attention if symptoms worsen',
                'Keep a record of all symptoms',
                'Contact healthcare provider'
            ],
            'recommendations': [
                'AI analysis unavailable - using basic assessment',
                'Consult medical professional for accurate diagnosis',
                'Emergency services: Call 911 for severe symptoms'
            ],
            'timestamp': datetime.now().isoformat(),
            'version': '2.0-Fallback'
        }

class ImageAnalyzer:
    """Computer vision for wound/rash analysis"""
    
    def __init__(self):
        self.skin_conditions = {
            'burn': {
                'characteristics': ['redness', 'blistering', 'peeling'],
                'severity_indicators': ['size', 'depth', 'location'],
                'first_aid': [
                    'Remove from heat source',
                    'Cool with water for 10-20 minutes',
                    'Do not break blisters',
                    'Cover with sterile bandage',
                    'Seek medical attention for severe burns'
                ]
            },
            'cut': {
                'characteristics': ['bleeding', 'open skin', 'straight edges'],
                'severity_indicators': ['depth', 'length', 'bleeding_amount'],
                'first_aid': [
                    'Apply direct pressure',
                    'Elevate if possible',
                    'Clean gently if minor',
                    'Apply bandage',
                    'Seek medical attention if deep or won\'t stop bleeding'
                ]
            },
            'rash': {
                'characteristics': ['red patches', 'bumps', 'irritation'],
                'severity_indicators': ['spreading', 'itching', 'swelling'],
                'first_aid': [
                    'Avoid scratching',
                    'Keep area clean and dry',
                    'Apply cool compress',
                    'Consider antihistamine for itching',
                    'Consult doctor if spreading or severe'
                ]
            }
        }
    
    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """Analyze uploaded image for medical conditions"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Basic image analysis
            analysis = self._analyze_skin_condition(cv_image)
            
            return analysis
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Image analysis failed: {str(e)}",
                'recommendations': [
                    'Please ensure image is clear and well-lit',
                    'Consult healthcare provider for visual examination',
                    'Take photo from appropriate distance'
                ]
            }
    
    def _analyze_skin_condition(self, image) -> Dict[str, Any]:
        """Analyze skin condition from image"""
        # Basic color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect redness (simplified approach)
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_percentage = np.sum(red_mask) / (red_mask.shape[0] * red_mask.shape[1] * 255) * 100
        
        # Simple classification based on redness
        if red_percentage > 15:
            condition = 'burn'
            severity = 'moderate' if red_percentage > 30 else 'mild'
        elif red_percentage > 5:
            condition = 'rash'
            severity = 'mild'
        else:
            condition = 'normal'
            severity = 'none'
        
        condition_data = self.skin_conditions.get(condition, {})
        
        return {
            'success': True,
            'detected_condition': condition,
            'severity': severity,
            'confidence': min(0.85, red_percentage / 20 + 0.3),
            'characteristics': condition_data.get('characteristics', []),
            'first_aid': condition_data.get('first_aid', []),
            'analysis_details': {
                'red_percentage': red_percentage,
                'image_quality': 'acceptable'
            },
            'recommendations': [
                f'Possible {condition} detected',
                'This is a basic AI analysis - consult medical professional',
                'Take additional photos if condition changes',
                'Seek immediate care if condition worsens'
            ]
        }

# Initialize AI engines
symptom_ai = AdvancedSymptomAnalyzer()
image_ai = ImageAnalyzer()