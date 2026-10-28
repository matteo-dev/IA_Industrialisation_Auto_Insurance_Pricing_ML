# Python file for auto insurance pricing model inference.

# Import required libraries
import joblib
import numpy as np
import os
from .processing import process_single_payload

# Define paths to locate the models_export directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models_export')

# Load preprocessors into memory
prep_cls = joblib.load(os.path.join(MODELS_DIR, 'prep_cls.joblib'))
prep_sev = joblib.load(os.path.join(MODELS_DIR, 'prep_sev.joblib'))

# Load models into memory
mod_freq_lr = joblib.load(os.path.join(MODELS_DIR, 'mod_freq_lr.joblib'))
mod_freq_xgb = joblib.load(os.path.join(MODELS_DIR, 'mod_freq_xgb.joblib'))
mod_sev_ridge = joblib.load(os.path.join(MODELS_DIR, 'mod_sev_ridge.joblib'))
mod_sev_xgb = joblib.load(os.path.join(MODELS_DIR, 'mod_sev_xgb.joblib'))

# Load actuarial parameters
params = joblib.load(os.path.join(MODELS_DIR, 'actuarial_params.joblib'))
w_freq = params['w_freq']
w_sev = params['w_sev']
sigma2_full = params['sigma2_full']
scale_ridge_full = params['scale_ridge_full']
scale_xgb_full = params['scale_xgb_full']

# Function to predict claim frequency for a policy
def predict_frequency(payload: dict) -> float:
    """Predicts claim probability for a policy."""
    df = process_single_payload(payload)
    X_p = prep_cls.transform(df)
    
    p_lr = mod_freq_lr.predict_proba(X_p)[:, 1][0]
    p_xgb = mod_freq_xgb.predict_proba(X_p)[:, 1][0]
    
    freq = w_freq * p_lr + (1 - w_freq) * p_xgb
    return max(0.0, float(freq))

# Function to predict claim severity if a claim occurs
def predict_amount(payload: dict) -> float:
    """Predicts claim amount if a claim occurs."""
    df = process_single_payload(payload)
    X_p = prep_sev.transform(df)
    
    # Ridge severity (with Duan smearing correction)
    pred_ridge_log = mod_sev_ridge.predict(X_p)[0]
    g_ridge = np.exp(pred_ridge_log + (sigma2_full / 2.0)) * scale_ridge_full
    
    # XGBoost severity
    g_xgb = mod_sev_xgb.predict(X_p)[0] * scale_xgb_full
    
    amount = w_sev * g_ridge + (1 - w_sev) * g_xgb
    return max(0.0, float(amount))

# Function to predict total pure premium following strict coursework rules
def predict_premium(payload: dict) -> dict:
    """Calculates total pure premium following strict coursework rules."""
    freq = predict_frequency(payload)
    amount = predict_amount(payload)
    
    # Coursework rule: premium = frequency * amount (non-negative)
    premium = max(0.0, float(freq * amount))
    
    return {
        "frequency": freq,
        "amount": amount,
        "premium": premium
    }