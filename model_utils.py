import os
import joblib

MODEL_PATH = os.path.join("models","symptom_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
        return _model
    
def predict_top_k(symptom_text, k=3):
    """
    Returns list of {"condition": <label>, "prob": <0..1>} sorted by prob desc
    """
    model = get_model()
    probs = model.predict_proba([symptom_text])[0]
    classes = model.named_steps['clf'].classes_
    pairs = list(zip(classes, probs))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    results = [{"condition": cls, "prob": float(prob)} for cls, prob in pairs_sorted[:k]]
    return results