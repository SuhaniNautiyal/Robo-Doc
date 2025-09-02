import os
import math
from flask import Blueprint, request, jsonify
from database import get_db

emergency_bp = Blueprint("emergency", __name__, url_prefix="/api/emergency")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@emergency_bp.route("/", methods=["POST"])
def emergency():
    data = request.json or {}
    lat = data.get("lat")
    lng = data.get("lng")
    top_n = int(data.get("top_n", 5))

    if lat is None or lng is None:
        return jsonify({"ok": False, "error": "lat and lng required"}), 400
    
    db = get_db()
    hospitals = list(db.hospitals.find({}, {"_id":0}))

    for h in hospitals:
        h["distance_km"] = haversine(float(lat), float(lng), float(h["lat"]), float(h["lng"]))

    hospitals_sorted = sorted(hospitals, key=lambda x: x["distance_km"])[:top_n]
    return jsonify({"ok": True, "hospitals": hospitals_sorted})
    
      

