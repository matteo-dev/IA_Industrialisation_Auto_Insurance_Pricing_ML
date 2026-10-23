# Fichier principal de l'application FastAPI pour l'API de tarification d'assurance automobile.

# Importation des bibliothèques nécessaires
from fastapi import FastAPI # Bibliothèque FastAPI pour créer l'API
from pydantic import BaseModel, Field # Bibliothèque Pydantic pour la validation des données
from typing import Optional # Bibliothèque typing pour les types optionnels
from .inference import predict_frequency, predict_amount, predict_premium

# Initialisation de l'API avec les chemins de documentation natifs
app = FastAPI(
    title="API Tarification Auto - M2 Fintech",
    description="API de tarification d'assurance automobile avec modèles ML industrialisés.",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# 1. Définition stricte du contrat de données (Schéma Pydantic)

# Classe ContractInput pour valider les données d'entrée de l'API
class ContractInput(BaseModel):
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

    # Les champs suivants peuvent valoir null selon le TD
    sex_conducteur2: Optional[str] = None
    anciennete_permis1: int
    anciennete_permis2: int
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

# 2. Définition des routes de l'API

# Route de santé pour vérifier si l'application est opérationnelle
@app.get("/health")
def health_check():
    """Vérifie l'état de santé de l'application."""
    return {"status": "ok"}

# Route pour prédire la fréquence des sinistres
@app.post("/predict_frequency")
def route_predict_frequency(contract: ContractInput):
    """Fait la prédiction du modèle de fréquence."""
    # model_dump() convertit l'objet Pydantic en dictionnaire pour notre processing
    freq = predict_frequency(contract.model_dump())
    return {"frequency": freq}

# Route pour prédire le montant des sinistres
@app.post("/predict_amount")
def route_predict_amount(contract: ContractInput):
    """Fait la prédiction du modèle de montant."""
    amount = predict_amount(contract.model_dump())
    return {"amount": amount}

# Route pour prédire la prime pure (fréquence * montant)
@app.post("/predict")
def route_predict(contract: ContractInput):
    """Fait la prédiction agrégée (Prime Pure = Fréquence * Montant)."""
    result = predict_premium(contract.model_dump())
    return result