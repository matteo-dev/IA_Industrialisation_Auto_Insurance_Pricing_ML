# ==============================================================================
# PROJET ASSURANCE AUTOMOBILE - PIPELINE FINAL "BEST OF BOTH WORLDS"
# Fusion Rigueur Actuarielle + ML Engineering + ecretement & Mutualisation
# ==============================================================================

# [Attribution des tâches - À modifier selon votre organisation reelle]
# - Emmanuel : Rigueur actuarielle (Capping, Mutualisation, Ridge Log)
# - Matteo : Pipeline ML, Feature Engineering, XGBoost, Modele de Blending
# ==============================================================================

# Importation des bibliotheques necessaires
import joblib # Bibliotheque pour la serialisation des modeles (sauvegarde et chargement)
import pandas as pd
import numpy as np 
import os
import shap # Bibliotheque pour l'explicabilite des modeles (SHAP)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold # Bibliotheque pour le split stratifie avec groupes (evite le data leakage)
from sklearn.compose import ColumnTransformer # Bibliotheque pour la transformation des colonnes (pretraitement)
from sklearn.pipeline import Pipeline # Bibliotheque pour la creation de pipelines de traitement (pretraitement + modele)
from sklearn.impute import SimpleImputer # Bibliotheque pour l'imputation des valeurs manquantes (mediane, mode, etc.)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder # Bibliotheque pour le pretraitement des donnees (scaling, encodage)
from sklearn.linear_model import LogisticRegression, Ridge # Bibliotheque pour les modeles lineaires (regression logistique et ridge)
from xgboost import XGBClassifier, XGBRegressor # Bibliotheque pour les modeles d'arbre de gradient boosting (XGBoost)
from scipy.optimize import minimize_scalar # Bibliotheque pour l'optimisation des poids de blending (minimisation)
from sklearn.metrics import root_mean_squared_error, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay # Bibliotheque pour l'evaluation des modeles (RMSE, AUC, ROC, matrice de confusion)
import warnings
import optuna # Bibliotheque pour l'optimisation des hyperparametres

# ==============================================================================
# CONFIGURATION INITIALE
# ==============================================================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
optuna.logging.set_verbosity(optuna.logging.WARNING) 

# Dossiers de sortie pour les resultats et les graphiques du PPT
OUTPUT_DIR = 'resultats_fusion'
PPT_DIR = 'visualisations_ppt'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PPT_DIR, exist_ok=True)

# Variables à exclure de l'apprentissage (Data Leakage ou IDs inutiles)
COLS_TO_DROP = [
    'nombre_sinistres', 'montant_sinistre', 'is_sinistre', 
    'id_client', 'id_vehicule', 'id_contrat', 'index', 
    'modele_vehicule', 'code_postal', 
    'debut_vente_vehicule', 'fin_vente_vehicule'
]

# ==============================================================================
# MODULE DE GeNeRATION DE GRAPHIQUES (POUR LE POWERPOINT)
# [Ajoute par Matteo / Emmanuel] : Outils de data viz pour la soutenance
# ==============================================================================

# Fonction pour generer les graphiques d'Analyse Exploratoire (EDA)
def generate_eda_plots(df):

    """Genere les graphiques d'Analyse Exploratoire (EDA) pour le PPT."""
    print("Generation des graphiques EDA...")
    
    # 1. Taux de sinistres (Desequilibre des classes)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_sinistre', data=df, palette='viridis')
    plt.title("Repartition des Sinistres (Desequilibre des classes)")
    plt.xlabel("Est Sinistre (0 = Non, 1 = Oui)")
    plt.ylabel("Nombre de contrats")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/01_desequilibre_classes.png", dpi=300)
    plt.close()

    # 2. Skewness de la gravite (Asymetrie des montants)
    df_sin = df[df['is_sinistre'] == 1]
    plt.figure(figsize=(8, 4))
    sns.histplot(df_sin['montant_sinistre'], bins=50, kde=True, color='red')
    plt.title("Distribution des montants de sinistres (Forte Asymetrie / Skewness)")
    plt.xlabel("Montant du sinistre (€)")
    plt.ylabel("Frequence")
    plt.xlim(0, df_sin['montant_sinistre'].quantile(0.95)) # Zoom sur 95% pour lisibilite
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/02_distribution_gravite_skewness.png", dpi=300)
    plt.close()

    # 3. Matrice de correlation (Variables numeriques)
    cols_num = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[cols_num].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title("Matrice de Correlation Lineaire")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/03_matrice_correlation.png", dpi=300)
    plt.close()

# Fonction pour generer les graphiques d'evaluation des modeles de classification (Frequence)
def plot_classification_metrics(y_true, y_prob_lr, y_prob_xgb, y_prob_blend):

    """Genere les graphiques d'evaluation du modele de Frequence."""
    print("Generation des graphiques de Classification (Frequence)...")
    
    # 1. Courbes ROC
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

    # 2. Matrice de confusion (basee sur le Blending)
    # Seuil fixe pour capturer au moins les 10% de profils les plus risques
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
# eTAPE 1 & 2 : PRePARATION DES DONNeES ET FEATURE ENGINEERING
# ==============================================================================

# Fonction pour regrouper les modalites rares d'une variable categorielle
def group_rare_modalities(df, col, threshold=50, fill_value='Autre'):

    """Regroupe les categories peu frequentes pour eviter le sur-apprentissage."""
    counts = df[col].value_counts()

    # On considere comme rare toute modalite ayant moins de 'threshold' occurrences
    rare_cats = counts[counts < threshold].index
    df[col] = df[col].replace(rare_cats, fill_value)
    return df

# Fonction de Feature Engineering (Creation de variables metiers intelligentes)
def feature_engineering(df, is_train=True):
    """
    [Conçu par Matteo] : Creation de variables metiers intelligentes.
    On cree des ratios et des flags pour aider les modeles arborescents et lineaires.
    """
    df_eng = df.copy()
    
    # Nettoyage des valeurs aberrantes
    for col in ['anciennete_permis1', 'anciennete_permis2']:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].replace(111, np.nan) # 111 est une erreur de saisie du dataset
            
    # Extraction d'informations geographiques (departement) à partir du code postal
    if 'code_postal' in df_eng.columns:
        df_eng['departement'] = df_eng['code_postal'].astype(str).str.zfill(5).str[:2]
        
    # Gestion des cardinalites elevees
    if 'marque_vehicule' in df_eng.columns:
        df_eng = group_rare_modalities(df_eng, 'marque_vehicule', threshold=50)

    # Variables croisees (Interactions metier)

    # Variable indicatrice de jeune conducteur (moins de 3 ans de permis) et calcul de l'age d'obtention du permis
    if 'age_conducteur1' in df_eng.columns and 'anciennete_permis1' in df_eng.columns:
        df_eng['age_obtention_permis1'] = df_eng['age_conducteur1'] - df_eng['anciennete_permis1']
        df_eng['is_jeune_permis'] = (df_eng['anciennete_permis1'] < 3).astype(int)
    
    # Variable de ratio puissance/poids
    if 'din_vehicule' in df_eng.columns and 'poids_vehicule' in df_eng.columns:
        df_eng['ratio_puissance_poids'] = df_eng['din_vehicule'] / (df_eng['poids_vehicule'] + 1)
    
    # Variable de ratio valeur/anciennete du vehicule
    if 'prix_vehicule' in df_eng.columns and 'anciennete_vehicule' in df_eng.columns:
        df_eng['ratio_valeur_anciennete'] = df_eng['prix_vehicule'] / (df_eng['anciennete_vehicule'] + 1)
    
    # Variable indicatrice de jeune conducteur2 (moins de 25 ans)
    if 'age_conducteur2' in df_eng.columns:
        df_eng['conducteur2_jeune'] = ((df_eng['age_conducteur2'] > 0) & (df_eng['age_conducteur2'] < 25)).astype(int)

    # Recapitulatif PPT (sauvegarde texte)
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

# Fonction pour construire un preprocesseur robuste (Pipeline scikit-learn)
def build_preprocessor(X_train):
    """
    [Conçu par Matteo] : Creation d'une pipeline scikit-learn robuste (Anti Data Leakage).
    Separe les traitements numeriques, categoriels et geographiques.
    """

    # Identification des variables numeriques, categoriels et geographiques
    var_num = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    var_cat = X_train.select_dtypes(include=['object', 'category', 'string']).columns.drop('departement', errors='ignore').tolist()
    var_geo = ['departement'] if 'departement' in X_train.columns else []

    # Numerique : Imputation par la mediane + Standardisation (Necessaire pour Ridge/LogReg)
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    
    # Categoriel : Imputation "Inconnu" + One-Hot Encoding (Ignore les nouvelles categories en test)
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Inconnu')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    # Geographique : Target Encoding (Lisse les moyennes par departement pour ne pas exploser la dimensionnalite)
    geo_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('target_enc', TargetEncoder(smooth='auto'))])

    # Construction du ColumnTransformer final
    transformers = [('num', num_pipe, var_num), ('cat', cat_pipe, var_cat)]
    if var_geo: transformers.append(('geo', geo_pipe, var_geo))
    
    return ColumnTransformer(transformers=transformers)

# ==============================================================================
# ETAPE 3 : MODeLISATION FReQUENCE (CLASSIFICATION)
# ==============================================================================

# Fonction pour entraîner les modeles de Frequence et trouver le blending optimal
def train_frequency(X_tr, y_tr, X_va, y_va):
    """
    [Collaboration Emmanuel / Matteo]
    Entraîne un modele lineaire simple et un arbre de gradient boosting,
    puis trouve la combinaison (blending) optimale.
    """
    print("\n--- Modelisation Frequence ---")
    
    # Modele 1 : Regression Logistique (Baseline robuste, bien calibree)
    logreg = LogisticRegression(max_iter=500, random_state=42)
    logreg.fit(X_tr, y_tr)
    
    # Recherche des meilleurs hyperparametres pour XGBClassifier
    print("Lancement de l'optimisation Optuna pour XGBClassifier...")

    # Fonction objectif pour Optuna (Maximiser l'AUC sur le jeu de validation)
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

        # Entraînement du modele XGBoost avec les hyperparametres proposes
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, preds)

    # Lancement de l'optimisation Optuna pour trouver les meilleurs hyperparametres
    study_freq = optuna.create_study(direction='maximize')
    study_freq.optimize(objective_freq, n_trials=10) 
    best_params_freq = study_freq.best_params
    best_params_freq.update({'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42})
    print(f"Meilleurs hyperparametres XGBoost Frequence : {best_params_freq}")

    # Modele 2 : XGBoost Classifier (Capture les effets non-lineaires complexes) avec Optuna
    xgb = XGBClassifier(**best_params_freq)
    xgb.fit(X_tr, y_tr)
    
    # Predictions sur le jeu de validation
    p_lr_va = logreg.predict_proba(X_va)[:, 1]
    p_xgb_va = xgb.predict_proba(X_va)[:, 1]
    
    # Optimisation du poids du blending pour maximiser l'AUC
    def obj(w):
        return -roc_auc_score(y_va, w * p_lr_va + (1 - w) * p_xgb_va)
    
    # Minimisation de la fonction objectif pour trouver le poids optimal
    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    w_opt = res.x
    print(f"Poids optimal Frequence : {w_opt:.2f} (LogReg) / {1-w_opt:.2f} (XGB)")
    
    # Generation des graphiques PPT pour la frequence
    p_blend_va = w_opt * p_lr_va + (1 - w_opt) * p_xgb_va
    plot_classification_metrics(y_va, p_lr_va, p_xgb_va, p_blend_va)
    
    return logreg, xgb, w_opt

# ==============================================================================
# ORCHESTRATION ET EXeCUTION PRINCIPALE
# ==============================================================================

# Fonction principale pour orchestrer le pipeline complet
if __name__ == "__main__":
    print("1. Chargement des donnees et Feature Engineering...")
    # On charge le jeu de donnees et on applique notre ingenierie metier
    df_train_raw = pd.read_csv('train.csv')
    df_train = feature_engineering(df_train_raw)
    
    # Creation de la cible binaire pour la frequence
    df_train['is_sinistre'] = (df_train['nombre_sinistres'] > 0).astype(int)
    
    # Generation des graphiques exploratoires initiaux
    generate_eda_plots(df_train)
    
    print("2. Split StratifiedGroupKFold (Validation rigoureuse)...")

    # GroupKFold pour eviter qu'un meme client soit dans le train ET le val (Data Leakage)
    cv = StratifiedGroupKFold(n_splits=5)
    train_idx, val_idx = next(cv.split(df_train, df_train['is_sinistre'], groups=df_train['id_client']))
    
    train_df, val_df = df_train.iloc[train_idx], df_train.iloc[val_idx]
    
    # Preparation donnees Frequence
    X_tr_cls = train_df.drop(columns=COLS_TO_DROP, errors='ignore')
    X_va_cls = val_df.drop(columns=COLS_TO_DROP, errors='ignore')
    y_tr_cls = train_df['is_sinistre']
    y_va_cls = val_df['is_sinistre']
    
    # Construction du preprocesseur pour la frequence
    prep_cls = build_preprocessor(X_tr_cls)
    X_tr_cls_p = prep_cls.fit_transform(X_tr_cls, y_tr_cls)
    X_va_cls_p = prep_cls.transform(X_va_cls)
    
    # Entraînement Modeles de Frequence
    mod_freq_lr, mod_freq_xgb, w_freq = train_frequency(X_tr_cls_p, y_tr_cls, X_va_cls_p, y_va_cls)

    # Preparation donnees Gravite (On filtre UNIQUEMENT sur les sinistres)
    train_sin = train_df[train_df['is_sinistre'] == 1]
    val_sin = val_df[val_df['is_sinistre'] == 1]
    
    # Construction du preprocesseur pour la gravite
    X_tr_sev = train_sin.drop(columns=COLS_TO_DROP, errors='ignore')
    X_va_sev = val_sin.drop(columns=COLS_TO_DROP, errors='ignore')
    y_tr_sev = train_sin['montant_sinistre']
    y_va_sev = val_sin['montant_sinistre']
    
    # Construction du preprocesseur pour la gravite
    prep_sev = build_preprocessor(X_tr_sev)
    X_tr_sev_p = prep_sev.fit_transform(X_tr_sev, y_tr_sev)
    X_va_sev_p = prep_sev.transform(X_va_sev) 
    X_va_sev_all_p = prep_sev.transform(X_va_cls) # Pour l'inference finale sur tout le dataset

    # ==============================================================================
    # eTAPE 4 : MODeLISATION DE LA GRAVITe AVEC eCReTEMENT (CAPPING)
    # ==============================================================================
    """
    [Conçu par Emmanuel] : La solution de l'industrie : L'ecretement et la Mutualisation.
    Les montants extremes detruisent l'apprentissage. On coupe les valeurs au 99e percentile
    lors de l'entraînement. Le modele apprend le "risque normal" (attritionnel).
    """
    print("\n--- Modelisation Gravite avec ecretement ---")

    # Calcul du seuil de capping (99e percentile) sur le jeu d'entraînement
    seuil_capping = np.percentile(y_tr_sev, 99)
    print(f"Seuil de capping (99e percentile) : {seuil_capping:.2f} €")
    
    # Capping uniquement sur l'entraînement
    y_tr_sev_capped = np.clip(y_tr_sev, a_min=None, a_max=seuil_capping)
    
    # Modele 1 Gravite : Ridge sur le Log de la cible

    # On utilise Ridge pour sa robustesse et sa capacite à gerer la multicolinearite.
    mod_sev_ridge = Ridge(alpha=50.0, random_state=42)

    # On prend le log de la cible pour stabiliser la variance et reduire l'impact des outliers
    y_tr_log_capped = np.log(y_tr_sev_capped)
    mod_sev_ridge.fit(X_tr_sev_p, y_tr_log_capped)
    pred_log_tr = mod_sev_ridge.predict(X_tr_sev_p)

    # Correction de Duan (quand on repasse du Log à l'exponentielle), cela permet de corriger le biais introduit par la transformation logarithmique
    sigma2_capped = np.var(y_tr_log_capped - pred_log_tr)
    scale_ridge_capped = y_tr_sev_capped.mean() / np.exp(pred_log_tr + (sigma2_capped / 2.0)).mean()
    
    # Optimisation des hyperparametres pour XGBRegressor (Gamma)
    print("Lancement de l'optimisation Optuna pour XGBRegressor...")

    # Fonction objectif pour Optuna (Minimiser le RMSE sur le jeu de validation)
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

        # Entraînement du modele XGBoost avec les hyperparametres proposes
        model = XGBRegressor(**params)
        model.fit(X_tr_sev_p, y_tr_sev_capped)
        
        # Prediction sur le jeu de validation et calcul du RMSE
        preds_va = model.predict(X_va_sev_p)
        scale_va = y_tr_sev_capped.mean() / model.predict(X_tr_sev_p).mean()
        return root_mean_squared_error(y_va_sev, preds_va * scale_va)
    
    # Lancement de l'optimisation Optuna pour trouver les meilleurs hyperparametres
    study_sev = optuna.create_study(direction='minimize')
    study_sev.optimize(objective_sev, n_trials=10) 
    best_params_sev = study_sev.best_params
    best_params_sev.update({'objective': 'reg:gamma', 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42})
    print(f"Meilleurs hyperparametres XGBoost Gravite : {best_params_sev}")

    # Modele 2 Gravite : XGBoost avec fonction de perte Gamma (special asymetrie) avec Optuna
    mod_sev_xgb = XGBRegressor(**best_params_sev)
    mod_sev_xgb.fit(X_tr_sev_p, y_tr_sev_capped)
    scale_xgb_capped = y_tr_sev_capped.mean() / mod_sev_xgb.predict(X_tr_sev_p).mean()
    
    # Blending Gravite sur le jeu de validation
    pred_ridge_va = np.exp(mod_sev_ridge.predict(X_va_sev_p) + (sigma2_capped / 2.0)) * scale_ridge_capped
    pred_xgb_va = mod_sev_xgb.predict(X_va_sev_p) * scale_xgb_capped
    
    # Fonction objectif pour minimiser le RMSE sur le jeu de validation
    def obj_sev(w):
        return root_mean_squared_error(y_va_sev, w * pred_ridge_va + (1 - w) * pred_xgb_va)
    
    # Optimisation du poids de blending pour la gravite
    w_sev = minimize_scalar(obj_sev, bounds=(0, 1), method='bounded').x
    print(f"Poids optimal Gravite : {w_sev:.2f} (Ridge) / {1-w_sev:.2f} (XGB)")

    # Graphique PPT Gravite
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
    # eTAPE 5 : CALCUL DE LA PRIME ET MUTUALISATION (CHARGEMENT)
    # ==============================================================================
    """
    [Conçu par Emmanuel] : Mutualisation (Le chargement).
    L'excedent coupe (tout ce qui depassait 99%) represente le coût des catastrophes.
    On calcule ce montant total et on le divise de maniere egale sur TOUS les assures.
    """
    print("\n--- Audit Financier sur le jeu de validation ---")
    
    # Calcul du chargement mutualise
    excedent_total = np.sum(y_tr_sev - y_tr_sev_capped)

    # Le chargement mutualise est le montant à ajouter à chaque prime pour couvrir les sinistres extremes
    chargement_mutualise = excedent_total / len(train_df)
    print(f"Excedent total mutualise : {excedent_total:,.2f} €")
    print(f"Chargement ajoute par assure (Mutualisation) : {chargement_mutualise:.2f} €")
    
    # 1. Prediction probabilite (Frequence)
    p_va = w_freq * mod_freq_lr.predict_proba(X_va_cls_p)[:, 1] + (1 - w_freq) * mod_freq_xgb.predict_proba(X_va_cls_p)[:, 1]
    
    # 2. Prediction gravite attritionnelle
    g_ridge_va_all = np.exp(mod_sev_ridge.predict(X_va_sev_all_p) + (sigma2_capped / 2.0)) * scale_ridge_capped
    g_xgb_va_all = mod_sev_xgb.predict(X_va_sev_all_p) * scale_xgb_capped
    g_va = w_sev * g_ridge_va_all + (1 - w_sev) * g_xgb_va_all
    
    # 3. Prime pure finale = (Freq * Grav) + Taxe de mutualisation
    prime_va = (p_va * g_va) + chargement_mutualise
    
    ## Audit Financier sur le jeu de validation
    primes_totales = prime_va.sum()
    sinistres_reels = val_df['montant_sinistre'].sum()
    rmse_val = root_mean_squared_error(val_df['montant_sinistre'], prime_va)
    
    # Affichage des resultats financiers et metriques
    print(f"Somme des primes percues : {primes_totales:,.2f} €")
    print(f"Somme des sinistres reels: {sinistres_reels:,.2f} €")
    print(f"Solde technique (P - S)  : {primes_totales - sinistres_reels:+,.2f} €")
    print(f"Loss Ratio (S/P)         : {(sinistres_reels/primes_totales)*100:.2f} %")
    print(f"RMSE (Metrique Kaggle)   : {rmse_val:.2f} €")

    # ==============================================================================
    # ReENTRAÎNEMENT GLOBAL ET SOUMISSION KAGGLE
    # ==============================================================================
    print("\n--- Reentraînement sur 100% des donnees et Soumission Kaggle ---")
    
    # Frequence Globale
    X_full_cls = df_train.drop(columns=COLS_TO_DROP, errors='ignore')
    y_full_cls = df_train['is_sinistre']
    prep_cls_full = build_preprocessor(X_full_cls)
    X_full_cls_p = prep_cls_full.fit_transform(X_full_cls, y_full_cls)
    
    mod_freq_lr.fit(X_full_cls_p, y_full_cls)
    mod_freq_xgb.fit(X_full_cls_p, y_full_cls)
    
    # Gravite Globale avec Capping
    df_sin_full = df_train[df_train['is_sinistre'] == 1]
    X_full_sev = df_sin_full.drop(columns=COLS_TO_DROP, errors='ignore')
    y_full_sev = df_sin_full['montant_sinistre']
    
    prep_sev_full = build_preprocessor(X_full_sev)
    X_full_sev_p = prep_sev_full.fit_transform(X_full_sev, y_full_sev)
    
    seuil_capping_full = np.percentile(y_full_sev, 99)
    y_full_sev_capped = np.clip(y_full_sev, a_min=None, a_max=seuil_capping_full)
    
    # Re-entraînement des modeles de gravite sur l'ensemble des donnees
    mod_sev_ridge.fit(X_full_sev_p, np.log(y_full_sev_capped))
    pred_log_full = mod_sev_ridge.predict(X_full_sev_p)
    sigma2_full = np.var(np.log(y_full_sev_capped) - pred_log_full)
    scale_ridge_full = y_full_sev_capped.mean() / np.exp(pred_log_full + (sigma2_full / 2.0)).mean()
    
    # Re-entraînement du modele XGBoost sur l'ensemble des donnees
    mod_sev_xgb.fit(X_full_sev_p, y_full_sev_capped)
    scale_xgb_full = y_full_sev_capped.mean() / mod_sev_xgb.predict(X_full_sev_p).mean()
    
    # Mutualisation Globale
    excedent_full = np.sum(y_full_sev - y_full_sev_capped)
    chargement_full = excedent_full / len(df_train)
    
    # Prediction sur Test.csv
    print("Generation des predictions sur test.csv...")
    try:
        # On charge le jeu de test et on applique le meme feature engineering
        df_test = feature_engineering(pd.read_csv('test.csv'), is_train=False)
        X_test = df_test.drop(columns=COLS_TO_DROP, errors='ignore')
        
        # On applique les memes transformations que pour l'entraînement
        X_test_cls_p = prep_cls_full.transform(X_test)
        X_test_sev_p = prep_sev_full.transform(X_test)
        
        # Predictions Frequence et Gravite sur le jeu de test
        p_test = w_freq * mod_freq_lr.predict_proba(X_test_cls_p)[:, 1] + (1 - w_freq) * mod_freq_xgb.predict_proba(X_test_cls_p)[:, 1]
        
        # Predictions Gravite sur le jeu de test
        g_ridge_test = np.exp(mod_sev_ridge.predict(X_test_sev_p) + (sigma2_full / 2.0)) * scale_ridge_full
        g_xgb_test = mod_sev_xgb.predict(X_test_sev_p) * scale_xgb_full
        g_test = w_sev * g_ridge_test + (1 - w_sev) * g_xgb_test
        
        # Calcul de la prime finale sur le jeu de test
        prime_test = (p_test * g_test) + chargement_full
        
        # Generation du fichier de soumission pour Kaggle
        submission = pd.DataFrame({'index': df_test['index'], 'pred': prime_test})
        submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_best_of_both.csv'), index=False)
        print(f"Fichier genere : {OUTPUT_DIR}/submission_best_of_both.csv")
    except FileNotFoundError:
        print("Fichier test.csv introuvable. Ignorer la soumission.")

    # ==============================================================================
    # EXPLICABILITe DU MODeLE AVEC SHAP (POUR LE PPT ET LE WORD)
    # ==============================================================================
    print("\n--- Generation des explications SHAP ---")
    
    # On utilise SHAP pour expliquer l'importance des variables dans le modele XGBoost de frequence
    explainer = shap.TreeExplainer(mod_freq_xgb)
    X_sample = X_va_cls_p[:1000]
    shap_values = explainer(X_sample)

    # Graphique SHAP Summary (Importance Globale et Impact des Variables)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=prep_cls.get_feature_names_out(), show=False)
    plt.title("Importance Globale et Impact des Variables (SHAP Summary)")
    plt.tight_layout()
    plt.savefig(f"{PPT_DIR}/08_shap_summary_plot.png", dpi=300)
    plt.close()
    
    print(f"Tous les graphiques PPT ont ete generes dans le dossier '{PPT_DIR}/'.")

# ==============================================================================
# Sauvegarde des modeles et preprocesseurs pour une utilisation future
# ==============================================================================

os.makedirs('../models_export', exist_ok=True)

# Sauvegarde des préprocesseurs fittés
joblib.dump(prep_cls_full, '../models_export/prep_cls.joblib')
joblib.dump(prep_sev_full, '../models_export/prep_sev.joblib')

# Sauvegarde des modèles entraînés
joblib.dump(mod_freq_lr, '../models_export/mod_freq_lr.joblib')
joblib.dump(mod_freq_xgb, '../models_export/mod_freq_xgb.joblib')
joblib.dump(mod_sev_ridge, '../models_export/mod_sev_ridge.joblib')
joblib.dump(mod_sev_xgb, '../models_export/mod_sev_xgb.joblib')

# Sauvegarde des scalaires de correction et poids
actuarial_params = {
    'w_freq': w_freq,
    'w_sev': w_sev,
    'sigma2_full': sigma2_full,
    'scale_ridge_full': scale_ridge_full,
    'scale_xgb_full': scale_xgb_full
}
joblib.dump(actuarial_params, '../models_export/actuarial_params.joblib')