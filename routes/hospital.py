import requests
from flask import Blueprint, request, jsonify
hospital_bp = Blueprint("hospital", __name__)

@hospital_bp.route("/nearest-hospitals", methods=["GET"])
def nearest_hospitals():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": "hospital",
        "format": "json",
        "limit": 5,
        "lat": lat,
        "lon": lon
    }
    headers = {
        "User-Agent": "RoboDoc/1.0 (contact@example.com)"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()

    except requests.RequestException as e:
        return jsonify({"error":str(e)}), 500
    
    hospitals = response.json()
    return jsonify(hospitals)