import os
import jwt
import datetime
from functools import wraps
from flask import request,jsonify
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY missing in .env")

JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "3600"))

def generate_jwt(payload: dict, exp_seconds: int=None) -> str:
    if exp_seconds is None:
        exp_seconds = JWT_EXP_SECONDS
    data = payload.copy()
    data["exp"] = datetime.datetime.utcnow() + datetime.timedelta(seconds=exp_seconds)
    token = jwt.encode(data, SECRET_KEY, algorithm="HS256")
    return token if isinstance(token, str) else token.decode()
def decode_jwt(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except Exception:
        return None
    
def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        parts = auth.split()
        if len(parts) != 2 or parts[0].lower()!= "bearer":
            return jsonify({"ok": False, "error": "Missing or invalid Authorization header"}), 401
        payload = decode_jwt(parts[1])
        if not payload:
            return jsonify({"ok": False, "error": "Invalid or expired token"}), 401
        request.jwt_payload = payload
        return fn(*args, **kwargs)
    return wrapper