from database import insert_many_symptoms, insert_hospitals
import json
import os

SYMPTOM_DOCS = [
    {
        "condition": "seizure",
        "keywords": ["seizure", "convulsion", "jerking"],
        "advice": (
            "1) Stay calm. 2) Clear nearby objects to prevent injury. 3) Cushion the head. " 
            "4) Do NOT restrain movements. 5) Turn the person on their side to keep airway clear. "
            "6) Call emergency services immediately. "
        )
    }, 
    {
        "condition": "heart_attack",
        "keywords": ["chest pain", "heart attack", "pressure chest"],
        "adivce": (
            "1) Call emergency services immediately. 2) Help the person sit and rest. "
            "3) Loosen tight clothing. 4) If available and trained, give asprin (300 mg) unless contraindicated."
            "5) Monitor breathing and pulse."
        )

    },
    {
        "condition": "stroke",
        "keywords": ["face droop", "slurred speech", "weakness", "stroke"],
        "advice": (
            "1) Call emergency services immediately (time matters). 2) Note time when symptoms started. "
            "3) Keep person calm and lying down with head slightly elevated. 4) Do not give food or drink."
        )
    },
    {
        "condition": "allergic_reaction",
        "keywords": ["hives", "swelling", "anaphylaxis", "allergy"],
        "advice":(
            "1) If person has an epipen, help them use it. 2) Call emergency services. "
            "3) Lay person down and raise legs if they are faint. 4) Monitor breathing. "
        )
    },
    {
        "condition": "severe_bleeding",
        "keywords": ["bleeding", "blood", "cut"],
        "advice": (
            "1) Apply firm pressure on the wound with a clean cloth. 2)Keep pressure until help arrives. "
            "3) If possible, elevate the injured part above heart level. 4) Call emergency services"
        )
    },
    {
        "condition": "choking",
        "keywords": ["choking", "cant breathe", "clutching throat"],
        "advice":(
            "1) If person can cough, encourage coughing. 2) If cannot breathe and you are trained perfrom Heimlich maneuver. "
            "3) Call emergency services immediately. "
        )
    },
    {
        "condition": "fainting",
        "keywords": [" fainting", " passed out", "lost consciousness"],
        "advice": (
            "1) Lay person flat and raise legs. 2) Check for breathing. 3) Loosen tight clothes. "
            "4) If not breathing, call emergency services and start CPR if trained."
        )
    },
    {
        "condition": "fracture",
        "keywords": ["fracture", "broken bone", "deformity"],
        "advice": (
            "1) Immobilize the limb (splint) and avoid moving the person unnecessarily. 2) Apply cold pack for swelling."
            "3). Get medical help for X-ray and treatment."
        )
    }
    
]

def seed():
    insert_many_symptoms(SYMPTOM_DOCS)

    hospitals_file = os.path.join("data","hospitals.json")
    if os.path.exists(hospitals_file):
        with open(hospitals_file, "r") as f:
            hospitals = json.load(f)
        insert_hospitals(hospitals)
    print("Seeding done. ")

if __name__ == "__main__":
    seed()
