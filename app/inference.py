# Fichier python pour l'inférence des modèles de tarification auto.

# Importation des bibliothèques nécessaires
import joblib
import numpy as np
import os
from .processing import process_single_payload

# Définition des chemins pour trouver le dossier models_export
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models_export')

# Chargement en mémoire des préprocesseurs
prep_cls = joblib.load(os.path.join(MODELS_DIR, 'prep_cls.joblib'))
prep_sev = joblib.load(os.path.join(MODELS_DIR, 'prep_sev.joblib'))

# Chargement en mémoire des modèles
mod_freq_lr = joblib.load(os.path.join(MODELS_DIR, 'mod_freq_lr.joblib'))
mod_freq_xgb = joblib.load(os.path.join(MODELS_DIR, 'mod_freq_xgb.joblib'))
mod_sev_ridge = joblib.load(os.path.join(MODELS_DIR, 'mod_sev_ridge.joblib'))
mod_sev_xgb = joblib.load(os.path.join(MODELS_DIR, 'mod_sev_xgb.joblib'))

# Chargement des paramètres actuariels
params = joblib.load(os.path.join(MODELS_DIR, 'actuarial_params.joblib'))
w_freq = params['w_freq']
w_sev = params['w_sev']
sigma2_full = params['sigma2_full']
scale_ridge_full = params['scale_ridge_full']
scale_xgb_full = params['scale_xgb_full']

# Fonction pour prédire la fréquence d'un sinistre pour un contrat
def predict_frequency(payload: dict) -> float:
    """Prédit la probabilité de sinistre pour un contrat."""
    df = process_single_payload(payload)
    X_p = prep_cls.transform(df)
    
    p_lr = mod_freq_lr.predict_proba(X_p)[:, 1][0]
    p_xgb = mod_freq_xgb.predict_proba(X_p)[:, 1][0]
    
    freq = w_freq * p_lr + (1 - w_freq) * p_xgb
    return max(0.0, float(freq))

# Fonction pour prédire le coût d'un sinistre s'il survient
def predict_amount(payload: dict) -> float:
    """Prédit le coût d'un sinistre s'il survient."""
    df = process_single_payload(payload)
    X_p = prep_sev.transform(df)
    
    # Gravité Ridge (avec correction de Duan)
    pred_ridge_log = mod_sev_ridge.predict(X_p)[0]
    g_ridge = np.exp(pred_ridge_log + (sigma2_full / 2.0)) * scale_ridge_full
    
    # Gravité XGBoost
    g_xgb = mod_sev_xgb.predict(X_p)[0] * scale_xgb_full
    
    amount = w_sev * g_ridge + (1 - w_sev) * g_xgb
    return max(0.0, float(amount))

# Fonction pour prédire la prime pure totale selon les règles strictes du TD
def predict_premium(payload: dict) -> dict:
    """Calcule la prime pure totale selon les règles strictes du TD."""
    freq = predict_frequency(payload)
    amount = predict_amount(payload)
    
    # Règle TD: premium = frequency * amount (positif ou nul)
    premium = max(0.0, float(freq * amount))
    
    return {
        "frequency": freq,
        "amount": amount,
        "premium": premium
    }