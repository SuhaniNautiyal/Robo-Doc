import os
import math
import json
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
from routes.hospital import hospital_bp
from routes.emergency import emergency_bp

from werkzeug.security import generate_password_hash, check_password_hash
import jwt

from database import init_db 
from model_utils import predict_top_k
from dotenv import load_dotenv

from auth_utils import generate_jwt, jwt_required
from encryption_utils import encrypt_string, decrypt_string
from flasgger import Swagger

app = Flask(__name__)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route":"/docs/"
}
swagger = Swagger(app, config=swagger_config)

app.register_blueprint(hospital_bp, url_prefix="/hospital")
app.register_blueprint(emergency_bp, url_prefix="/emergency")

init_db()
@app.route('/')
def home():
    """
    RoboDoc Home
    ---
    responses:
     200:
      description : Welcome message
    """
    return jsonify({"message":"Welcome to RoboDoc API with Swagger!"})

if __name__ == '__main__':
    app.run(debug=True)
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "3600"))

app = Flask(__name__)
CORS(app)

app.register_blueprint(hospital_bp, url_prefix="/api")
app.register_blueprint(emergency_bp, url_prefix="/api")
def create_user(email, password):
    db = init_db()
    hashed = generate_password_hash(password)
    if db.users.find_one({"email": email}):
        return False, "User exists"
    db.users.insert_one({"email": email, "password": hashed})
    return True, "User created"

def check_user_and_get_token(email, password):
    db = init_db()
    user = db.users.find_one({"email": email})
    if not user:
        return None
    if check_password_hash(user["password"], password):
        payload = {
            "email": email,
            "exp": datetime.utcnow() + timedelta(seconds=JWT_EXP_SECONDS)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return token
    
    return None

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        return None
    
@app.route("/")
def hello():
    return "RoboDoc API (Phase 1) is running."
    
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    emergency_contact = data.get("emergency_contact")
    if not email or not password:
        return jsonify({"ok": False, "error": "email and password required"}), 400
    db = init_db()
    if db.users.find_one({"email": email}):
        return jsonify({"ok": False, "error": "User exists"}), 400
    
    hashed = generate_password_hash(password)
    enc_contact = encrypt_string(emergency_contact) if emergency_contact else None

    db.users.insert_one({
        "email": email,
        "password": hashed,
        "emergency_contact": enc_contact 
    })
    return json({"ok": True, "message": "User created"}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"ok": False, "error": "email and password required"}), 400
    
    db = init_db()
    user = db.users.find_one({"email":email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"ok": False, "error": "invalid credentials"}), 401
    token = generate_jwt({"email": email})
    return jsonify({"ok": True, "token": token})

@app.route("/api/check_symptoms", methods=["POST"])
def check_symptoms():
    data = request.json or {}
    symptoms_text = data.get("symptoms", "")
    if not symptoms_text:
        return jsonify({"ok": False, "error": "symptoms field required"}), 400
    
    try:
        preds = predict_top_k(symptoms_text, k=3)
    except Exception as e:
        return jsonify({"ok": False, "error": f"model error: {str(e)}"}), 500
    
    db = init_db()
    results = []
    for p in preds:
        cond = p["condition"]
        doc = db.symptoms.find_one({"condition": cond}, {"_id": 0})
        advice = doc["advice"] if doc and "advice" in doc else "No advice found for this condition. "
        results.append({
            "condition": cond,
            "probability": p["prob"],
            "advice": advice
        })

    disclaimer = (
        "This is an automated assistant for preliminary guidance only - not a substitute for professional medical care. "
        "If symptoms are severe or life threatening, call emergency services immediately. "

    )
    return jsonify({"ok": True, "predictions": results, "disclaimer": disclaimer})

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

@app.route("/api/emergency", methods=["POST"])
def emergency():
    data = request.json or {}
    lat = data.get("lat")
    lng = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"ok": False, "error": "lat and lng required"}), 400
    
    db = init_db()
    hospitals = list(db.hospitals.find({}, {"_id":0}))
    for h in hospitals:
        h["distance_km"] = haversine(float(lat), float(lng), float(h["lat"]), float(h["lng"]))

    hospitals_sorted = sorted(hospitals, key=lambda x : x["distance_km"])[:5]
    return jsonify({"ok": True, "hospitals": hospitals_sorted})

if __name__ == "__main__":
    print(app.url_map)
    print("Starting RoboDoc Phase 1 API on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
    
@app.route("/api/user/me", methods=["GET"])
@jwt_required
def me():
    payload = request.jwt_payload
    email = payload.get("email")
    db = init_db()
    user = db.users.find_one({"email": email})
    if not user:
        return jsonify({"ok": False, "error":"user not found"}), 404
    
    response = {
        "email": user.get("email"),
        "emergency_contact": decrypt_string(user.get("emergency_contact")) if user.get("emergency_contact") else None

    }
    return jsonify({"ok": True, "user": response})