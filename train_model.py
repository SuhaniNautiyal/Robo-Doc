import os
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas
DATA = [
    #seizure
    ("uncontrolled shaking, loss of consciousness, jerking movements, foam at mouth", "seizure"),
    ("body convulsions, drooling, sudden collapse, stiff body", "seizure"),
    ("sudden jerks, not responsive, eyes rolling, twitching", "seizure"),

    #heart attack
    ("chest pain, pressure in chest, radiating pain to left arm, sweating", "heart_attack"),
    ("tightness in chest, shortness of breath, nausea, dizziness", "heart_attack"),
    ("severe chest discomfort and breathlessness", "heart_attack"),

    #stroke
    ("sudden weakness on one side, face droop, slurred speech, difficulty speaking", "stroke"),
    ("cant move one arm, mouth drooping, confusion, trouble walking", "stroke"),

    #Allergic reaction/ Anaphylaxis
    ("hives, swelling of face and throat, difficulty breathing, itchy skin", "allergic_reaction"),
    ("sudden swelling, breathing difficulty, rash and faintness", "allergic_reaction"),
    
    #Severe bleeding
    ("heavy bleeding from wound, continuous bleeding, blood spurting", "severe_bleeding"),
    ("deep cut, blood wont stop, pulsing blood flow", "severe_bleeding"),

    #choking
    ("cant breathe, clutching throat, coughing and high-pitched noise, cant speak", "choking"),
    ("unable to breathe or cough, hands at throat, wheeze sound", "choking"),

    #Fainting
    ("sudden fainting, feeling lightheaded, brief loss of consciousness", "fainting"),
    ("dizzy then passed out for a short time, pale skin", "fainting"),

    #fracture
    ("severe pain after fall, deformity of limb, unable to move arm or leg", "fracture"),
    ("swelling and bruising after injury, bone looks out of place", "fracture")
]

texts = [t for t, label in DATA]
labels = [label for t, label in DATA]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=tokenize_and_lemmatize, lowercase=True, token_pattern=None)),
    ("clf", LogisticRegression(max_iter=1000))
])

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "symptom_model.pkl")
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
