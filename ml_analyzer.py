# RoboDoc Phase 2 - Advanced ML Symptom Analyzer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import datetime
from typing import Dict, List, Any, Tuple

class AdvancedMLAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.is_trained = False
        
        # Enhanced medical knowledge base with ML training data
        self.training_data = self._create_training_data()
        self.emergency_keywords = [
            'chest pain', 'heart attack', 'stroke', 'seizure', 'unconscious', 
            'not breathing', 'choking', 'severe bleeding', 'cardiac arrest',
            'difficulty breathing', 'severe burns', 'poisoning', 'overdose',
            'head injury', 'spine injury', 'broken bones', 'severe allergic reaction',
            'high fever', 'severe pain', 'cant breathe', 'passed out'
        ]
        
        # Risk categories with detailed classification
        self.risk_categories = {
            'CRITICAL': {
                'color': '#FF0000',
                'priority': 1,
                'response_time': '<2 minutes',
                'action': 'Call 112 IMMEDIATELY'
            },
            'HIGH': {
                'color': '#FF6B35', 
                'priority': 2,
                'response_time': '<30 minutes',
                'action': 'Seek emergency care now'
            },
            'MEDIUM': {
                'color': '#F7931E',
                'priority': 3, 
                'response_time': '<24 hours',
                'action': 'Visit doctor today'
            },
            'LOW': {
                'color': '#4CAF50',
                'priority': 4,
                'response_time': '<7 days', 
                'action': 'Monitor and consult if needed'
            }
        }
        
        # Train the model
        self._train_model()
    
    def _create_training_data(self) -> List[Dict]:
        """Create comprehensive training dataset for Indian medical scenarios"""
        return [
            # CRITICAL Emergency Cases
            {"symptoms": "chest pain crushing feeling left arm numb", "risk": "CRITICAL", "confidence": 0.95},
            {"symptoms": "not breathing unconscious blue lips", "risk": "CRITICAL", "confidence": 0.98},
            {"symptoms": "severe bleeding wont stop losing consciousness", "risk": "CRITICAL", "confidence": 0.92},
            {"symptoms": "seizure convulsions shaking uncontrollable", "risk": "CRITICAL", "confidence": 0.90},
            {"symptoms": "heart attack chest crushing pain sweating", "risk": "CRITICAL", "confidence": 0.95},
            {"symptoms": "stroke face drooping speech slurred weakness", "risk": "CRITICAL", "confidence": 0.93},
            {"symptoms": "choking cant breathe object stuck throat", "risk": "CRITICAL", "confidence": 0.96},
            {"symptoms": "severe allergic reaction swelling throat breathing difficulty", "risk": "CRITICAL", "confidence": 0.94},
            {"symptoms": "poisoning vomiting severely unresponsive", "risk": "CRITICAL", "confidence": 0.91},
            {"symptoms": "head injury unconscious bleeding skull", "risk": "CRITICAL", "confidence": 0.93},
            
            # HIGH Risk Cases
            {"symptoms": "high fever 104 degrees delirium", "risk": "HIGH", "confidence": 0.85},
            {"symptoms": "severe abdominal pain vomiting blood", "risk": "HIGH", "confidence": 0.87},
            {"symptoms": "broken bone visible deformity severe pain", "risk": "HIGH", "confidence": 0.82},
            {"symptoms": "severe burns third degree large area", "risk": "HIGH", "confidence": 0.88},
            {"symptoms": "difficulty breathing asthma attack wheezing", "risk": "HIGH", "confidence": 0.84},
            {"symptoms": "severe headache worst ever sudden onset", "risk": "HIGH", "confidence": 0.86},
            {"symptoms": "snake bite swelling spreading rapidly", "risk": "HIGH", "confidence": 0.89},
            {"symptoms": "severe dehydration heat stroke confusion", "risk": "HIGH", "confidence": 0.83},
            {"symptoms": "kidney stones severe pain nausea", "risk": "HIGH", "confidence": 0.81},
            {"symptoms": "appendicitis severe right side pain fever", "risk": "HIGH", "confidence": 0.85},
            
            # MEDIUM Risk Cases  
            {"symptoms": "fever 101 body ache cough", "risk": "MEDIUM", "confidence": 0.75},
            {"symptoms": "persistent cough two weeks blood", "risk": "MEDIUM", "confidence": 0.78},
            {"symptoms": "sprain ankle swollen painful walking", "risk": "MEDIUM", "confidence": 0.70},
            {"symptoms": "food poisoning diarrhea vomiting mild fever", "risk": "MEDIUM", "confidence": 0.73},
            {"symptoms": "urinary tract infection burning sensation fever", "risk": "MEDIUM", "confidence": 0.72},
            {"symptoms": "migraine severe headache nausea light sensitivity", "risk": "MEDIUM", "confidence": 0.76},
            {"symptoms": "gastritis stomach pain bloating acid reflux", "risk": "MEDIUM", "confidence": 0.71},
            {"symptoms": "allergic reaction rash itching mild swelling", "risk": "MEDIUM", "confidence": 0.74},
            {"symptoms": "back pain muscle strain lifting heavy", "risk": "MEDIUM", "confidence": 0.69},
            {"symptoms": "ear infection pain drainage hearing loss", "risk": "MEDIUM", "confidence": 0.73},
            
            # LOW Risk Cases
            {"symptoms": "common cold runny nose sneezing", "risk": "LOW", "confidence": 0.60},
            {"symptoms": "minor cut small wound bleeding stopped", "risk": "LOW", "confidence": 0.65},
            {"symptoms": "headache stress tired long day", "risk": "LOW", "confidence": 0.58},
            {"symptoms": "muscle soreness after exercise workout", "risk": "LOW", "confidence": 0.62},
            {"symptoms": "mild nausea after eating spicy food", "risk": "LOW", "confidence": 0.55},
            {"symptoms": "dry cough throat irritation dust", "risk": "LOW", "confidence": 0.59},
            {"symptoms": "minor bruise small bump pain", "risk": "LOW", "confidence": 0.63},
            {"symptoms": "insomnia trouble sleeping stress", "risk": "LOW", "confidence": 0.57},
            {"symptoms": "mild indigestion bloated after meal", "risk": "LOW", "confidence": 0.61},
            {"symptoms": "seasonal allergies sneezing watery eyes", "risk": "LOW", "confidence": 0.64}
        ]
    
    def _train_model(self):
        """Train the machine learning model with symptom data"""
        try:
            # Prepare training data
            symptoms = [data['symptoms'] for data in self.training_data]
            risks = [data['risk'] for data in self.training_data]
            
            # Create feature vectors
            X = self.vectorizer.fit_transform(symptoms)
            y = risks
            
            # Train the model
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Calculate accuracy
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            predictions = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            print(f"🧠 ML Model trained successfully! Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
            self.is_trained = False
    
    def analyze_symptoms_advanced(self, symptoms: str, patient_age: int = None, gender: str = None) -> Dict[str, Any]:
        """Advanced symptom analysis using ML + rule-based approach"""
        
        # Clean and preprocess symptoms
        symptoms_clean = self._preprocess_symptoms(symptoms)
        
        # Get ML prediction
        ml_result = self._get_ml_prediction(symptoms_clean)
        
        # Get rule-based analysis  
        rule_result = self._rule_based_analysis(symptoms_clean)
        
        # Combine results intelligently
        final_result = self._combine_analyses(ml_result, rule_result, symptoms_clean, patient_age, gender)
        
        # Add emergency detection
        final_result['emergency_detected'] = self._detect_emergency(symptoms_clean)
        
        # Add severity score (1-10)
        final_result['severity_score'] = self._calculate_severity_score(final_result)
        
        # Add first aid instructions
        final_result['first_aid_steps'] = self._get_first_aid_instructions(final_result['riskLevel'])
        
        # Add follow-up recommendations
        final_result['recommendations'] = self._get_recommendations(final_result)
        
        return final_result
    
    def _preprocess_symptoms(self, symptoms: str) -> str:
        """Clean and standardize symptom text"""
        # Convert to lowercase
        symptoms = symptoms.lower().strip()
        
        # Remove extra spaces
        symptoms = re.sub(r'\s+', ' ', symptoms)
        
        # Handle common misspellings
        corrections = {
            'stomache': 'stomach',
            'feaver': 'fever', 
            'headacke': 'headache',
            'throath': 'throat',
            'cheast': 'chest',
            'nausious': 'nauseous'
        }
        
        for wrong, correct in corrections.items():
            symptoms = symptoms.replace(wrong, correct)
        
        return symptoms
    
    def _get_ml_prediction(self, symptoms: str) -> Dict[str, Any]:
        """Get ML model prediction"""
        if not self.is_trained:
            return {"riskLevel": "MEDIUM", "confidence": 0.5}
        
        try:
            # Vectorize symptoms
            X = self.vectorizer.transform([symptoms])
            
            # Get prediction and probabilities
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Get confidence (max probability)
            max_prob_idx = np.argmax(probabilities)
            confidence = probabilities[max_prob_idx]
            
            return {
                "riskLevel": prediction,
                "confidence": confidence,
                "probabilities": dict(zip(self.classifier.classes_, probabilities))
            }
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return {"riskLevel": "MEDIUM", "confidence": 0.5}
    
    def _rule_based_analysis(self, symptoms: str) -> Dict[str, Any]:
        """Traditional rule-based symptom analysis"""
        symptoms_lower = symptoms.lower()
        
        # Critical emergency keywords
        critical_keywords = [
            'chest pain', 'heart attack', 'stroke', 'seizure', 'unconscious',
            'not breathing', 'choking', 'severe bleeding', 'cardiac arrest'
        ]
        
        # High priority keywords
        high_keywords = [
            'high fever', 'severe pain', 'difficulty breathing', 'severe headache',
            'broken bone', 'severe burns', 'poisoning', 'snake bite'
        ]
        
        # Medium priority keywords
        medium_keywords = [
            'fever', 'cough', 'vomiting', 'diarrhea', 'headache', 'nausea',
            'abdominal pain', 'back pain', 'sprain'
        ]
        
        # Check for matches
        critical_matches = sum(1 for keyword in critical_keywords if keyword in symptoms_lower)
        high_matches = sum(1 for keyword in high_keywords if keyword in symptoms_lower)
        medium_matches = sum(1 for keyword in medium_keywords if keyword in symptoms_lower)
        
        if critical_matches > 0:
            return {"riskLevel": "CRITICAL", "confidence": 0.9, "matches": critical_matches}
        elif high_matches > 0:
            return {"riskLevel": "HIGH", "confidence": 0.8, "matches": high_matches}
        elif medium_matches > 0:
            return {"riskLevel": "MEDIUM", "confidence": 0.7, "matches": medium_matches}
        else:
            return {"riskLevel": "LOW", "confidence": 0.6, "matches": 0}
    
    def _combine_analyses(self, ml_result: Dict, rule_result: Dict, symptoms: str, age: int, gender: str) -> Dict[str, Any]:
        """Intelligently combine ML and rule-based results"""
        
        # Priority mapping
        priority_map = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        
        # Get priorities
        ml_priority = priority_map.get(ml_result["riskLevel"], 2)
        rule_priority = priority_map.get(rule_result["riskLevel"], 2)
        
        # Choose higher priority (more conservative approach for medical)
        final_priority = max(ml_priority, rule_priority)
        
        # Map back to risk level
        priority_to_risk = {4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"}
        final_risk = priority_to_risk[final_priority]
        
        # Calculate combined confidence
        ml_confidence = ml_result.get("confidence", 0.5)
        rule_confidence = rule_result.get("confidence", 0.5)
        combined_confidence = (ml_confidence + rule_confidence) / 2
        
        # Age-based adjustments
        if age:
            if age > 65 or age < 2:
                # Higher risk for elderly and infants
                if final_risk == "LOW":
                    final_risk = "MEDIUM"
                elif final_risk == "MEDIUM":
                    final_risk = "HIGH"
                combined_confidence = min(combined_confidence + 0.1, 1.0)
        
        return {
            "riskLevel": final_risk,
            "confidence": round(combined_confidence, 2),
            "ml_prediction": ml_result["riskLevel"],
            "rule_prediction": rule_result["riskLevel"],
            "risk_category": self.risk_categories[final_risk],
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
    
    def _detect_emergency(self, symptoms: str) -> bool:
        """Detect if this is a true emergency requiring immediate attention"""
        emergency_indicators = [
            'chest pain', 'heart attack', 'stroke', 'seizure', 'unconscious',
            'not breathing', 'choking', 'severe bleeding', 'cardiac arrest',
            'cant breathe', 'passed out', 'severe burns', 'poisoning'
        ]
        
        symptoms_lower = symptoms.lower()
        return any(indicator in symptoms_lower for indicator in emergency_indicators)
    
    def _calculate_severity_score(self, analysis: Dict) -> int:
        """Calculate severity score from 1-10"""
        risk_to_score = {
            "CRITICAL": 9,
            "HIGH": 7, 
            "MEDIUM": 4,
            "LOW": 2
        }
        
        base_score = risk_to_score.get(analysis["riskLevel"], 2)
        confidence = analysis.get("confidence", 0.5)
        
        # Adjust based on confidence
        adjusted_score = int(base_score + (confidence - 0.5) * 2)
        
        return max(1, min(10, adjusted_score))
    
    def _get_first_aid_instructions(self, riskLevel: str) -> List[str]:
        """Get appropriate first aid instructions based on risk level"""
        instructions = {
            "CRITICAL": [
                "🚨 CALL 112 IMMEDIATELY - DO NOT DELAY",
                "Keep person calm and comfortable",
                "Monitor breathing and pulse constantly", 
                "If trained, prepare to perform CPR",
                "Clear airway if choking",
                "Control bleeding with direct pressure",
                "Do not give food or water",
                "Stay with person until help arrives"
            ],
            "HIGH": [
                "📞 Call 108 for ambulance",
                "Keep person comfortable and still",
                "Monitor vital signs closely",
                "Apply basic first aid as needed",
                "Do not move if spinal injury suspected",
                "Keep warm but not hot",
                "Note all symptoms and timing"
            ],
            "MEDIUM": [
                "🏥 Schedule doctor visit today",
                "Rest and avoid strenuous activity",
                "Stay hydrated with small sips",
                "Monitor symptoms for changes",
                "Take temperature every 2 hours",
                "Apply ice for swelling, heat for pain"
            ],
            "LOW": [
                "🏠 Rest and self-care",
                "Monitor symptoms for 24-48 hours",
                "Stay hydrated and get adequate sleep",
                "Use over-the-counter medications as needed",
                "Contact doctor if symptoms worsen"
            ]
        }
        
        return instructions.get(riskLevel, instructions["MEDIUM"])
    
    def _get_recommendations(self, analysis: Dict) -> List[str]:
        """Get personalized recommendations"""
        riskLevel = analysis["riskLevel"]
        
        recommendations = {
            "CRITICAL": [
                "This requires IMMEDIATE emergency care",
                "Do not drive yourself - call ambulance", 
                "Have someone stay with you at all times",
                "Bring list of medications to hospital"
            ],
            "HIGH": [
                "Seek medical attention within 4-6 hours",
                "Do not ignore these symptoms",
                "Have emergency contacts ready",
                "Consider urgent care or ER visit"
            ],
            "MEDIUM": [
                "Schedule appointment with your doctor",
                "Monitor symptoms for next 24-48 hours",
                "Keep a symptom diary",
                "Call doctor if symptoms worsen"
            ],
            "LOW": [
                "Continue monitoring your symptoms",
                "Practice good self-care",
                "Consider telemedicine consultation",
                "See doctor if symptoms persist beyond a week"
            ]
        }
        
        return recommendations.get(riskLevel, recommendations["MEDIUM"])
    def analyze(self, symptoms: str, patient_age: int = None, gender: str = None) -> Dict[str, Any]:
        """
        Simple wrapper so Flask can call ml_analyzer.analyze(symptoms).
        Internally uses analyze_symptoms_advanced.
        """
        return self.analyze_symptoms_advanced(symptoms, patient_age, gender)

# Global analyzer instance
ml_analyzer = AdvancedMLAnalyzer()