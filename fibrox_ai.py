import pickle
import xgboost as xgb
import numpy as np
np.random.seed(42)

# FibroX AI-specific features and logic
MODEL_PATH = "xgboost_model_isF3_Youden_Index.pkl"
MODEL_FEATURES = [
    'Age (years)',
    'Glycohemoglobin (%)',
    'Alanine aminotransferase (U/L)',
    'Aspartate aminotransferase (U/L)',
    'Platelet count (1000 cells/µL)',
    'Body-mass index (kg/m**2)',
    'GFR_EPI'
]

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

def calculate_gfr(serum_cr, age, is_female):
    """
    Calculate GFR using the CKD-EPI equation.
    
    Args:
        serum_cr: Serum creatinine in mg/dL
        age: Age in years  
        is_female: Boolean indicating if patient is female
    
    Returns:
        Calculated GFR value
    """
    if is_female:
        if serum_cr <= 0.7:
            A = 0.7
            B = -0.241
        else:
            A = 0.7 
            B = -1.2
    else:
        if serum_cr <= 0.9:
            A = 0.9
            B = -0.302
        else:
            A = 0.9
            B = -1.2
            
    gfr = 142 * ((serum_cr/A)**B) * (0.9938**age)
    
    if is_female:
        gfr *= 1.012
        
    return gfr

def predict(model, data):
    threshold = 0.54  # Custom threshold
    data_dmatrix = xgb.DMatrix(data[MODEL_FEATURES])
    y_pred_proba = model.predict(data_dmatrix)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return y_pred, y_pred_proba
