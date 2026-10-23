# Fichier principal de l'API FastAPI pour la tarification automobile. Il définit les routes, les schémas de données et la documentation Swagger.

# Importation des bibliothèques nécessaires
from fastapi import FastAPI # Bibliothèque principale pour créer l'API REST
from pydantic import BaseModel, Field # Bibliothèque pour la validation stricte des données d'entrée
from typing import Optional 
import uvicorn # Bibliothèque pour lancer le serveur ASGI (Uvicorn)

# Import des fonctions d'inférence depuis le module local
from .inference import predict_frequency, predict_amount, predict_premium

# 1. METADONNÉES ET CONFIGURATION SWAGGER (OPENAPI)
tags_metadata = [
    {
        "name": "Monitoring",
        "description": "Sondes de vérification de l'état du serveur.",
    },
    {
        "name": "Moteur Actuariel",
        "description": "Endpoints d'inférence de Machine Learning pour la tarification.",
    },
]

app = FastAPI(
    title="🚗 API Tarification Auto - M2 Fintech",
    description="""
    **API REST industrialisée** pour la tarification de contrats d'assurance automobile.
    
    ### Modélisation :
    * **Fréquence** : Blending optimal (Régression Logistique + XGBClassifier)
    * **Gravité** : Blending optimal (Ridge Log-transformé + XGBRegressor Gamma) sur le 99e percentile
    * **Prime Pure** : Application stricte de la formule `Fréquence * Montant`
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"syntaxHighlight.theme": "monokai"}
)

# 2. CONTRAT DE DONNÉES STRICT (SCHÉMA PYDANTIC)

# Classe ContractInput pour le dataset de base
class ContractInput(BaseModel):
    """
    Schéma strict d'un contrat d'assurance. 
    Les types sont vérifiés automatiquement. Si un champ manque ou est mal typé,
    l'API renverra une erreur 422 Unprocessable Entity.
    """
    id_client: str
    id_vehicule: str
    id_contrat: str
    bonus: float
    type_contrat: str
    duree_contrat: int
    anciennete_info: int
    freq_paiement: str
    paiement: str
    utilisation: str
    code_postal: str
    conducteur2: str
    age_conducteur1: int
    age_conducteur2: int
    sex_conducteur1: str
    
    # Les champs suivants peuvent valoir null (None en Python)
    sex_conducteur2: Optional[str] = None
    
    anciennete_permis1: int
    anciennete_permis2: int
    
    # Peut valoir null selon les règles du TD
    anciennete_vehicule: Optional[int] = None
    
    cylindre_vehicule: int
    din_vehicule: int
    essence_vehicule: str
    marque_vehicule: str
    modele_vehicule: str
    debut_vente_vehicule: int
    fin_vente_vehicule: int
    vitesse_vehicule: int
    type_vehicule: str
    prix_vehicule: float
    poids_vehicule: int

# 3. DÉFINITION DES ROUTES DE L'API

@app.get("/health", tags=["Monitoring"], summary="Vérifier la santé de l'API")
def health_check():
    """
    Sonde de viabilité. Renvoie un code 200 et un statut 'ok' si le serveur
    est fonctionnel et prêt à accepter des connexions.
    """
    return {"status": "ok"}

@app.post("/predict_frequency", tags=["Moteur Actuariel"], summary="Prédire la fréquence")
def route_predict_frequency(contract: ContractInput):
    """
    Calcule la probabilité de survenance d'un sinistre pour un contrat donné 
    via le blending (LogReg + XGBClassifier).
    Renvoie un flottant positif ou nul.
    """
    # .model_dump() convertit le payload Pydantic validé en dictionnaire Python
    freq = predict_frequency(contract.model_dump())
    return {"frequency": max(0.0, float(freq))}

@app.post("/predict_amount", tags=["Moteur Actuariel"], summary="Prédire la gravité")
def route_predict_amount(contract: ContractInput):
    """
    Estime le coût du sinistre s'il venait à se produire via le blending 
    (Ridge Log avec correction Duan + XGBRegressor Gamma).
    Renvoie un flottant positif ou nul.
    """
    amount = predict_amount(contract.model_dump())
    return {"amount": max(0.0, float(amount))}

@app.post("/predict", tags=["Moteur Actuariel"], summary="Calculer la prime pure totale")
def route_predict(contract: ContractInput):
    """
    Agrège la fréquence et la gravité pour retourner la prime pure totale 
    selon la règle actuarielle stricte du TD : premium = frequency * amount.
    """
    payload = contract.model_dump()
    
    # Appel de ton fichier d'inférence
    result = predict_premium(payload)
    
    return {
        "frequency": max(0.0, float(result["frequency"])),
        "amount": max(0.0, float(result["amount"])),
        "premium": max(0.0, float(result["premium"]))
    }

# ==============================================================================
# POINT D'ENTRÉE POUR LANCEMENT DIRECT (OPTIONNEL)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)