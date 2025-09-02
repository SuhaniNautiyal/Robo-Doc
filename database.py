import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI =os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

_client = None
def get_client():
    global _client
    if not MONGO_URI:
           raise ValueError(" MONGO_URI is missing. Check your .env file.")
    if _client is None:
        try:
               _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
               _client.server_info()
        except Exception as e:
               raise ConnectionError(f"Failed to connect to MongoDB: {e}")
        
    return _client
    
def get_db():
        return get_client()[DB_NAME]
    
def get_symptom_advice(condition):
        db = get_db()
        doc = db.symptoms.find_one({"condition": condition}, {"_id": 0})
        return doc
    
def insert_many_symptoms(docs):
        db = get_db()
        db.symptoms.delete_many({})
        db.symptoms.insert_many(docs)

def insert_hospitals(hospitals):
            db = get_db()
            db.hospitals.delete_many({})
            db.hospitals.insert_many(hospitals)