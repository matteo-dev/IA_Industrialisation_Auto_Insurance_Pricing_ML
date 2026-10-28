# ==============================================================================
# AUTO INSURANCE PROJECT - FINAL PIPELINE "BEST OF BOTH WORLDS"
# Merging Actuarial Rigor + ML Engineering + Capping & Risk Pooling
# ==============================================================================

# Import required libraries
import joblib # Library for model serialization (saving and loading)
import pandas as pd
import numpy as np 
import os
import shap # Library for model explainability (SHAP)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold # Library for stratified group split (prevents data leakage)
from sklearn.compose import ColumnTransformer # Library for column transformation (preprocessing)
from sklearn.pipeline import Pipeline # Library to build processing pipelines (preprocessing + model)
from sklearn.impute import SimpleImputer # Library for missing value imputation (median, mode, etc.)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder # Library for data preprocessing (scaling, encoding)
from sklearn.linear_model import LogisticRegression, Ridge # Library for linear models (logistic regression and ridge)
from xgboost import XGBClassifier, XGBRegressor # Library for gradient boosted tree models (XGBoost)
from scipy.optimize import minimize_scalar # Library for blending weights optimization (minimization)
from sklearn.metrics import root_mean_squared_error, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay # Library for model evaluation (RMSE, AUC, ROC, confusion matrix)
import warnings
import optuna # Library for hyperparameter optimization

# ==============================================================================
# INITIAL CONFIGURATION
# ==============================================================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
optuna.logging.set_verbosity(optuna.logging.WARNING) 

# Output directories for results and presentation plots
OUTPUT_DIR = 'resultats_fusion'
PPT_DIR = 'visualisations_ppt'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PPT_DIR, exist_ok=True)

# Features to exclude from training (Data Leakage or redundant IDs)
COLS_TO_DROP = [
    'nombre_sinistres', 'montant_sinistre', 'is_sinistre', 
    'id_client', 'id_vehicule', 'id_contrat', 'index', 
    'modele_vehicule', 'code_postal', 
    'debut_vente_vehicule', 'fin_vente_vehicule'
]

# ==============================================================================
# PLOT GENERATION MODULE (FOR POWERPOINT PRESENTATION)
# [Added by Matteo / Emmanuel]: Data visualization tools for the defense presentation
# ==============================================================================

# Function to generate Exploratory Data Analysis (EDA) plots
def generate_eda_plots(df):

    """Generates Exploratory Data Analysis (EDA) plots for the presentation."""
    print("Generation des graphiques EDA...")
    
    # 1. Claim rate (Class imbalance)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_sinistre', data=df, palette='viridis')
    plt.title("Repartition des Sinistres (Desequilibre des classes)")
    plt.xlabel("Est Sinistre (0 = Non, 1 = Oui)")
    plt.ylabel("Nombre de contrats")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/01_desequilibre_classes.png", dpi=300)
    plt.close()

    # 2. Severity skewness (Asymmetry of claim amounts)
    df_sin = df[df['is_sinistre'] == 1]
    plt.figure(figsize=(8, 4))
    sns.histplot(df_sin['montant_sinistre'], bins=50, kde=True, color='red')
    plt.title("Distribution des montants de sinistres (Forte Asymetrie / Skewness)")
    plt.xlabel("Montant du sinistre (€)")
    plt.ylabel("Frequence")
    plt.xlim(0, df_sin['montant_sinistre'].quantile(0.95)) # Zoom on 95th percentile for readability
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/02_distribution_gravite_skewness.png", dpi=300)
    plt.close()

    # 3. Correlation matrix (Numerical features)
    cols_num = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[cols_num].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title("Matrice de Correlation Lineaire")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/03_matrice_correlation.png", dpi=300)
    plt.close()

# Function to generate evaluation plots for classification models (Frequency)
def plot_classification_metrics(y_true, y_prob_lr, y_prob_xgb, y_prob_blend):

    """Generates evaluation plots for the Frequency model."""
    print("Generation des graphiques de Classification (Frequence)...")
    
    # 1. ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_prob_xgb)
    fpr_blend, tpr_blend, _ = roc_curve(y_true, y_prob_blend)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Reg Log (AUC = {roc_auc_score(y_true, y_prob_lr):.3f})', linestyle='--')
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_true, y_prob_xgb):.3f})')
    plt.plot(fpr_blend, tpr_blend, label=f'Blending (AUC = {roc_auc_score(y_true, y_prob_blend):.3f})', linewidth=2, color='black')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Courbe ROC - Modeles de Frequence")
    plt.xlabel("Taux de Faux Positifs")
    plt.ylabel("Taux de Vrais Positifs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/04_roc_auc_frequence.png", dpi=300)
    plt.close()

    # 2. Confusion matrix (based on Blending)
    # Fixed threshold to capture at least the top 10% riskiest profiles
    seuil = np.percentile(y_prob_blend, 90) 
    y_pred_class = (y_prob_blend >= seuil).astype(int)
    cm = confusion_matrix(y_true, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non Sinistre', 'Sinistre'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Matrice de Confusion (Seuil top 10% risque: {seuil:.3f})")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/05_matrice_confusion.png", dpi=300)
    plt.close()

# ==============================================================================
# STEPS 1 & 2: DATA PREPARATION AND FEATURE ENGINEERING
# ==============================================================================

# Function to group rare categories of a categorical feature
def group_rare_modalities(df, col, threshold=50, fill_value='Autre'):

    """Groups infrequent categories to prevent overfitting."""
    counts = df[col].value_counts()

    # Any category with fewer occurrences than 'threshold' is considered rare
    rare_cats = counts[counts < threshold].index
    df[col] = df[col].replace(rare_cats, fill_value)
    return df

# Feature Engineering function (Creation of domain-specific features)
def feature_engineering(df, is_train=True):
    """
    [Designed by Matteo]: Creation of domain-specific features.
    Computes ratios and indicator flags to assist tree-based and linear models.
    """
    df_eng = df.copy()
    
    # Clean anomalous entries
    for col in ['anciennete_permis1', 'anciennete_permis2']:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].replace(111, np.nan) # 111 is an input data entry error
            
    # Extract geographical information (department) from postal code
    if 'code_postal' in df_eng.columns:
        df_eng['departement'] = df_eng['code_postal'].astype(str).str.zfill(5).str[:2]
        
    # Manage high cardinality
    if 'marque_vehicule' in df_eng.columns:
        df_eng = group_rare_modalities(df_eng, 'marque_vehicule', threshold=50)

    # Interaction terms (Domain-specific interactions)

    # Young driver indicator flag (less than 3 years of license) and license acquisition age
    if 'age_conducteur1' in df_eng.columns and 'anciennete_permis1' in df_eng.columns:
        df_eng['age_obtention_permis1'] = df_eng['age_conducteur1'] - df_eng['anciennete_permis1']
        df_eng['is_jeune_permis'] = (df_eng['anciennete_permis1'] < 3).astype(int)
    
    # Power-to-weight ratio feature
    if 'din_vehicule' in df_eng.columns and 'poids_vehicule' in df_eng.columns:
        df_eng['ratio_puissance_poids'] = df_eng['din_vehicule'] / (df_eng['poids_vehicule'] + 1)
    
    # Vehicle value-to-age ratio feature
    if 'prix_vehicule' in df_eng.columns and 'anciennete_vehicule' in df_eng.columns:
        df_eng['ratio_valeur_anciennete'] = df_eng['prix_vehicule'] / (df_eng['anciennete_vehicule'] + 1)
    
    # Secondary young driver indicator flag (under 25 years old)
    if 'age_conducteur2' in df_eng.columns:
        df_eng['conducteur2_jeune'] = ((df_eng['age_conducteur2'] > 0) & (df_eng['age_conducteur2'] < 25)).astype(int)

    # Presentation summary (plain text save)
    if is_train:
        with open(f"{PPT_DIR}/06_recap_feature_engineering.txt", "w") as f:
            f.write("Nouvelles variables creees :\n")
            f.write("- departement (extrait du code postal)\n")
            f.write("- age_obtention_permis1\n")
            f.write("- is_jeune_permis\n")
            f.write("- ratio_puissance_poids\n")
            f.write("- ratio_valeur_anciennete\n")
            f.write("- conducteur2_jeune\n")

    return df_eng

# Function to build a robust preprocessor (scikit-learn Pipeline)
def build_preprocessor(X_train):
    """
    [Designed by Matteo]: Creates a robust scikit-learn pipeline (prevents data leakage).
    Separates numeric, categorical, and geographical processing.
    """

    # Identify numeric, categorical, and geographical columns
    var_num = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    var_cat = X_train.select_dtypes(include=['object', 'category', 'string']).columns.drop('departement', errors='ignore').tolist()
    var_geo = ['departement'] if 'departement' in X_train.columns else []

    # Numerical: Median imputation + Standardization (Required for Ridge/LogReg)
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    
    # Categorical: "Unknown" imputation + One-Hot Encoding (Ignores unseen categories in test)
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Inconnu')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    # Geographical: Target Encoding (Smooths departmental averages to avoid dimensional explosion)
    geo_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('target_enc', TargetEncoder(smooth='auto'))])

    # Build final ColumnTransformer
    transformers = [('num', num_pipe, var_num), ('cat', cat_pipe, var_cat)]
    if var_geo: transformers.append(('geo', geo_pipe, var_geo))
    
    return ColumnTransformer(transformers=transformers)

# ==============================================================================
# STEP 3: FREQUENCY MODELING (CLASSIFICATION)
# ==============================================================================

# Function to train Frequency models and determine optimal blending
def train_frequency(X_tr, y_tr, X_va, y_va):
    """
    [Collaboration Emmanuel / Matteo]
    Trains a simple linear baseline and a gradient boosted decision tree,
    then solves for the optimal blending weights.
    """
    print("\n--- Modelisation Frequence ---")
    
    # Model 1: Logistic Regression (Robust, well-calibrated baseline)
    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(X_tr, y_tr)
    
    # Hyperparameter search for XGBClassifier
    print("Lancement de l'optimisation Optuna pour XGBClassifier...")

    # Objective function for Optuna (Maximize AUC on validation set)
    def objective_freq(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42
        }

        # Train XGBoost model with candidate hyperparameters
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, preds)

    # Run Optuna optimization study to find best hyperparameters
    study_freq = optuna.create_study(direction='maximize')
    study_freq.optimize(objective_freq, n_trials=10) 
    best_params_freq = study_freq.best_params
    best_params_freq.update({'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42})
    print(f"Meilleurs hyperparametres XGBoost Frequence : {best_params_freq}")

    # Model 2: XGBoost Classifier (Captures complex non-linear patterns) with Optuna
    xgb = XGBClassifier(**best_params_freq)
    xgb.fit(X_tr, y_tr)
    
    # Predictions on validation set
    p_lr_va = logreg.predict_proba(X_va)[:, 1]
    p_xgb_va = xgb.predict_proba(X_va)[:, 1]
    
    # Optimize blending weight to maximize AUC
    def obj(w):
        return -roc_auc_score(y_va, w * p_lr_va + (1 - w) * p_xgb_va)
    
    # Minimize objective function to compute optimal weight
    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    w_opt = res.x
    print(f"Poids optimal Frequence : {w_opt:.2f} (LogReg) / {1-w_opt:.2f} (XGB)")
    
    # Generate presentation plots for frequency
    p_blend_va = w_opt * p_lr_va + (1 - w_opt) * p_xgb_va
    plot_classification_metrics(y_va, p_lr_va, p_xgb_va, p_blend_va)
    
    return logreg, xgb, w_opt

# ==============================================================================
# PIPELINE ORCHESTRATION AND MAIN EXECUTION
# ==============================================================================

# Main entry point to run full pipeline
if __name__ == "__main__":
    print("1. Chargement des donnees et Feature Engineering...")
    # Load dataset and apply domain feature engineering
    df_train_raw = pd.read_csv('train.csv')
    df_train = feature_engineering(df_train_raw)
    
    # Create binary target for claim frequency
    df_train['is_sinistre'] = (df_train['nombre_sinistres'] > 0).astype(int)
    
    # Generate initial exploratory plots
    generate_eda_plots(df_train)
    
    print("2. Split StratifiedGroupKFold (Validation rigoureuse)...")

    # GroupKFold to prevent data leakage from same policyholder across train/val
    cv = StratifiedGroupKFold(n_splits=5)
    train_idx, val_idx = next(cv.split(df_train, df_train['is_sinistre'], groups=df_train['id_client']))
    
    train_df, val_df = df_train.iloc[train_idx], df_train.iloc[val_idx]
    
    # Prepare frequency dataset
    X_tr_cls = train_df.drop(columns=COLS_TO_DROP, errors='ignore')
    X_va_cls = val_df.drop(columns=COLS_TO_DROP, errors='ignore')
    y_tr_cls = train_df['is_sinistre']
    y_va_cls = val_df['is_sinistre']
    
    # Build frequency preprocessor
    prep_cls = build_preprocessor(X_tr_cls)
    X_tr_cls_p = prep_cls.fit_transform(X_tr_cls, y_tr_cls)
    X_va_cls_p = prep_cls.transform(X_va_cls)
    
    # Train Frequency Models
    mod_freq_lr, mod_freq_xgb, w_freq = train_frequency(X_tr_cls_p, y_tr_cls, X_va_cls_p, y_va_cls)

    # Prepare severity dataset (filter ONLY on actual claims)
    train_sin = train_df[train_df['is_sinistre'] == 1]
    val_sin = val_df[val_df['is_sinistre'] == 1]
    
    # Prepare severity feature matrices
    X_tr_sev = train_sin.drop(columns=COLS_TO_DROP, errors='ignore')
    X_va_sev = val_sin.drop(columns=COLS_TO_DROP, errors='ignore')
    y_tr_sev = train_sin['montant_sinistre']
    y_va_sev = val_sin['montant_sinistre']
    
    # Build severity preprocessor
    prep_sev = build_preprocessor(X_tr_sev)
    X_tr_sev_p = prep_sev.fit_transform(X_tr_sev, y_tr_sev)
    X_va_sev_p = prep_sev.transform(X_va_sev) 
    X_va_sev_all_p = prep_sev.transform(X_va_cls) # For final inference across entire dataset

    # ==============================================================================
    # STEP 4: SEVERITY MODELING WITH CAPPING (Ecretement)
    # ==============================================================================
    """
    [Designed by Emmanuel]: Industry-standard solution: Capping and Risk Pooling.
    Extreme outliers disrupt model convergence. Values are clipped at the 99th percentile
    during training so the model learns the standard ("attritional") risk baseline.
    """
    print("\n--- Modelisation Gravite avec ecretement ---")

    # Compute capping threshold (99th percentile) on training set
    seuil_capping = np.percentile(y_tr_sev, 99)
    print(f"Seuil de capping (99e percentile) : {seuil_capping:.2f} €")
    
    # Apply capping exclusively to training set
    y_tr_sev_capped = np.clip(y_tr_sev, a_min=None, a_max=seuil_capping)
    
    # Severity Model 1: Ridge regression on log-transformed target

    # Ridge regularizes against multicollinearity and maintains robustness.
    mod_sev_ridge = Ridge(alpha=50.0, random_state=42)

    # Take logarithm of target to stabilize variance and curb outlier sensitivity
    y_tr_log_capped = np.log(y_tr_sev_capped)
    mod_sev_ridge.fit(X_tr_sev_p, y_tr_log_capped)
    pred_log_tr = mod_sev_ridge.predict(X_tr_sev_p)

    # Duan smearing retransformation correction factor (log-to-exponential transformation bias correction)
    sigma2_capped = np.var(y_tr_log_capped - pred_log_tr)
    scale_ridge_capped = y_tr_sev_capped.mean() / np.exp(pred_log_tr + (sigma2_capped / 2.0)).mean()
    
    # Hyperparameter optimization for XGBRegressor (Gamma objective)
    print("Lancement de l'optimisation Optuna pour XGBRegressor...")

    # Objective function for Optuna (Minimize RMSE on validation set)
    def objective_sev(trial):
        params = {
            'objective': 'reg:gamma',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': 42
        }

        # Train XGBoost model with proposed hyperparameters
        model = XGBRegressor(**params)
        model.fit(X_tr_sev_p, y_tr_sev_capped)
        
        # Predict on validation set and evaluate RMSE
        preds_va = model.predict(X_va_sev_p)
        scale_va = y_tr_sev_capped.mean() / model.predict(X_tr_sev_p).mean()
        return root_mean_squared_error(y_va_sev, preds_va * scale_va)
    
    # Run Optuna optimization study to find best hyperparameters
    study_sev = optuna.create_study(direction='minimize')
    study_sev.optimize(objective_sev, n_trials=10) 
    best_params_sev = study_sev.best_params
    best_params_sev.update({'objective': 'reg:gamma', 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42})
    print(f"Meilleurs hyperparametres XGBoost Gravite : {best_params_sev}")

    # Severity Model 2: XGBoost with Gamma loss function (skewness handling) via Optuna
    mod_sev_xgb = XGBRegressor(**best_params_sev)
    mod_sev_xgb.fit(X_tr_sev_p, y_tr_sev_capped)
    scale_xgb_capped = y_tr_sev_capped.mean() / mod_sev_xgb.predict(X_tr_sev_p).mean()
    
    # Severity Blending on validation set
    pred_ridge_va = np.exp(mod_sev_ridge.predict(X_va_sev_p) + (sigma2_capped / 2.0)) * scale_ridge_capped
    pred_xgb_va = mod_sev_xgb.predict(X_va_sev_p) * scale_xgb_capped
    
    # Objective function to minimize RMSE on validation set
    def obj_sev(w):
        return root_mean_squared_error(y_va_sev, w * pred_ridge_va + (1 - w) * pred_xgb_va)
    
    # Optimize blending weight for severity
    w_sev = minimize_scalar(obj_sev, bounds=(0, 1), method='bounded').x
    print(f"Poids optimal Gravite : {w_sev:.2f} (Ridge) / {1-w_sev:.2f} (XGB)")

    # Severity presentation plot
    rmse_ridge = root_mean_squared_error(y_va_sev, pred_ridge_va)
    rmse_xgb = root_mean_squared_error(y_va_sev, pred_xgb_va)
    rmse_blend = root_mean_squared_error(y_va_sev, w_sev * pred_ridge_va + (1 - w_sev) * pred_xgb_va)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Ridge (Log)', 'XGBoost (Gamma)', 'Blending'], y=[rmse_ridge, rmse_xgb, rmse_blend], palette='magma')
    plt.title("Comparaison RMSE - Modeles de Gravite")
    plt.ylabel("RMSE (€)")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/07_rmse_gravite.png", dpi=300)
    plt.close()

    # ==============================================================================
    # STEP 5: PURE PREMIUM CALCULATION AND RISK POOLING (Loading)
    # ==============================================================================
    """
    [Designed by Emmanuel]: Mutualization (Risk Loading).
    The capped excess (everything above 99th percentile) covers catastrophic claims cost.
    This pooled aggregate amount is distributed equally across ALL policyholders.
    """
    print("\n--- Audit Financier sur le jeu de validation ---")
    
    # Calculate pooled mutualization surcharge
    excedent_total = np.sum(y_tr_sev - y_tr_sev_capped)

    # Mutualized loading is added to each premium to cover tail risk and catastrophic losses
    chargement_mutualise = excedent_total / len(train_df)
    print(f"Excedent total mutualise : {excedent_total:,.2f} €")
    print(f"Chargement ajoute par assure (Mutualisation) : {chargement_mutualise:.2f} €")
    
    # 1. Frequency probability prediction
    p_va = w_freq * mod_freq_lr.predict_proba(X_va_cls_p)[:, 1] + (1 - w_freq) * mod_freq_xgb.predict_proba(X_va_cls_p)[:, 1]
    
    # 2. Attritional severity prediction
    g_ridge_va_all = np.exp(mod_sev_ridge.predict(X_va_sev_all_p) + (sigma2_capped / 2.0)) * scale_ridge_capped
    g_xgb_va_all = mod_sev_xgb.predict(X_va_sev_all_p) * scale_xgb_capped
    g_va = w_sev * g_ridge_va_all + (1 - w_sev) * g_xgb_va_all
    
    # 3. Final pure premium = (Freq * Sev) + Mutualization loading
    prime_va = (p_va * g_va) + chargement_mutualise
    
    ## Financial audit on validation set
    primes_totales = prime_va.sum()
    sinistres_reels = val_df['montant_sinistre'].sum()
    rmse_val = root_mean_squared_error(val_df['montant_sinistre'], prime_va)
    
    # Display financial metrics and evaluation results
    print(f"Somme des primes percues : {primes_totales:,.2f} €")
    print(f"Somme des sinistres reels: {sinistres_reels:,.2f} €")
    print(f"Solde technique (P - S)  : {primes_totales - sinistres_reels:+,.2f} €")
    print(f"Loss Ratio (S/P)         : {(sinistres_reels/primes_totales)*100:.2f} %")
    print(f"RMSE (Metrique Kaggle)   : {rmse_val:.2f} €")

    # ==============================================================================
    # FULL RETRAINING AND KAGGLE SUBMISSION
    # ==============================================================================
    print("\n--- Reentraînement sur 100% des donnees et Soumission Kaggle ---")
    
    # Full dataset Frequency
    X_full_cls = df_train.drop(columns=COLS_TO_DROP, errors='ignore')
    y_full_cls = df_train['is_sinistre']
    prep_cls_full = build_preprocessor(X_full_cls)
    X_full_cls_p = prep_cls_full.fit_transform(X_full_cls, y_full_cls)
    
    mod_freq_lr.fit(X_full_cls_p, y_full_cls)
    mod_freq_xgb.fit(X_full_cls_p, y_full_cls)
    
    # Full dataset Severity with Capping
    df_sin_full = df_train[df_train['is_sinistre'] == 1]
    X_full_sev = df_sin_full.drop(columns=COLS_TO_DROP, errors='ignore')
    y_full_sev = df_sin_full['montant_sinistre']
    
    prep_sev_full = build_preprocessor(X_full_sev)
    X_full_sev_p = prep_sev_full.fit_transform(X_full_sev, y_full_sev)
    
    seuil_capping_full = np.percentile(y_full_sev, 99)
    y_full_sev_capped = np.clip(y_full_sev, a_min=None, a_max=seuil_capping_full)
    
    # Retrain severity models across all available training data
    mod_sev_ridge.fit(X_full_sev_p, np.log(y_full_sev_capped))
    pred_log_full = mod_sev_ridge.predict(X_full_sev_p)
    sigma2_full = np.var(np.log(y_full_sev_capped) - pred_log_full)
    scale_ridge_full = y_full_sev_capped.mean() / np.exp(pred_log_full + (sigma2_full / 2.0)).mean()
    
    # Retrain XGBoost severity model across full dataset
    mod_sev_xgb.fit(X_full_sev_p, y_full_sev_capped)
    scale_xgb_full = y_full_sev_capped.mean() / mod_sev_xgb.predict(X_full_sev_p).mean()
    
    # Global Mutualization calculation
    excedent_full = np.sum(y_full_sev - y_full_sev_capped)
    chargement_full = excedent_full / len(df_train)
    
    # Test.csv inference
    print("Generation des predictions sur test.csv...")
    try:
        # Load test set and apply matching feature engineering
        df_test = feature_engineering(pd.read_csv('test.csv'), is_train=False)
        X_test = df_test.drop(columns=COLS_TO_DROP, errors='ignore')
        
        # Apply fitted preprocessing pipelines
        X_test_cls_p = prep_cls_full.transform(X_test)
        X_test_sev_p = prep_sev_full.transform(X_test)
        
        # Frequency and Severity predictions on test set
        p_test = w_freq * mod_freq_lr.predict_proba(X_test_cls_p)[:, 1] + (1 - w_freq) * mod_freq_xgb.predict_proba(X_test_cls_p)[:, 1]
        
        # Severity predictions on test set
        g_ridge_test = np.exp(mod_sev_ridge.predict(X_test_sev_p) + (sigma2_full / 2.0)) * scale_ridge_full
        g_xgb_test = mod_sev_xgb.predict(X_test_sev_p) * scale_xgb_full
        g_test = w_sev * g_ridge_test + (1 - w_sev) * g_xgb_test
        
        # Compute final pure premium on test set
        prime_test = (p_test * g_test) + chargement_full
        
        # Generate submission file for Kaggle
        submission = pd.DataFrame({'index': df_test['index'], 'pred': prime_test})
        submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_best_of_both.csv'), index=False)
        print(f"Fichier genere : {OUTPUT_DIR}/submission_best_of_both.csv")
    except FileNotFoundError:
        print("Fichier test.csv introuvable. Ignorer la soumission.")

    # ==============================================================================
    # MODEL EXPLAINABILITY WITH SHAP (FOR PRESENTATION AND REPORT)
    # ==============================================================================
    print("\n--- Generation des explications SHAP ---")
    
    # Use SHAP to explain feature importance for frequency XGBoost model
    explainer = shap.TreeExplainer(mod_freq_xgb)
    X_sample = X_va_cls_p[:1000]
    shap_values = explainer(X_sample)

    # SHAP Summary Plot (Global Feature Importance and Directional Impact)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=prep_cls.get_feature_names_out(), show=False)
    plt.title("Importance Globale et Impact des Variables (SHAP Summary)")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/08_shap_summary_plot.png", dpi=300)
    plt.close()
    
    print(f"Tous les graphiques PPT ont ete generes dans le dossier '{PPT_DIR}/'.")

# ==============================================================================
# Model and preprocessor serialization for downstream deployment
# ==============================================================================

os.makedirs('../models_export', exist_ok=True)

# Save fitted preprocessors
joblib.dump(prep_cls_full, '../models_export/prep_cls.joblib')
joblib.dump(prep_sev_full, '../models_export/prep_sev.joblib')

# Save trained models
joblib.dump(mod_freq_lr, '../models_export/mod_freq_lr.joblib')
joblib.dump(mod_freq_xgb, '../models_export/mod_freq_xgb.joblib')
joblib.dump(mod_sev_ridge, '../models_export/mod_sev_ridge.joblib')
joblib.dump(mod_sev_xgb, '../models_export/mod_sev_xgb.joblib')

# Save correction scale factors and blending weights
actuarial_params = {
    'w_freq': w_freq,
    'w_sev': w_sev,
    'sigma2_full': sigma2_full,
    'scale_ridge_full': scale_ridge_full,
    'scale_xgb_full': scale_xgb_full
}
joblib.dump(actuarial_params, '../models_export/actuarial_params.joblib')