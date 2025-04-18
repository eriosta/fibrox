import pickle
import numpy as np
np.random.seed(42)

# At-Risk AI-specific features and logic
MODEL_PATH = "atrisk35.pkl"
MODEL_FEATURES = [
    'ALANINE AMINOTRANSFERASE (ALT) (U/L)', 
    'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 
    'PLATELET COUNT (1000 CELLS/UL)', 
    'AGE IN YEARS AT SCREENING', 
    'BODY MASS INDEX (KG/M**2)'
]

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

def predict(model, data):
    y_pred_proba = model.predict_proba(data)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Default threshold 0.5
    return y_pred, y_pred_proba
