import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()

FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise RuntimeError("FERNET_KEY not set in .env")

fernet = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

def encrypt_string(plain_text: str) -> bytes:
    if plain_text is None:
        return None
    if isinstance(plain_text, str):
        plain_text = plain_text.encode()
    return fernet.encrypt(plain_text)

def decrypt_string(token: bytes) ->str:
    if token is None:
        return None
    try:
        return fernet.decrypt(token).decode()
    except Exception:
        return None