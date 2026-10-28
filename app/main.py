# Main FastAPI file for auto insurance pricing. Defines routes, data schemas, and Swagger documentation.

# Import required libraries
from fastapi import FastAPI # Core library to build the REST API
from pydantic import BaseModel, Field # Library for strict input data validation
from typing import Optional 
import uvicorn # Library to run the ASGI server (Uvicorn)

# Import inference functions from local module
from .inference import predict_frequency, predict_amount, predict_premium

# 1. METADATA AND SWAGGER CONFIGURATION (OPENAPI)
tags_metadata = [
    {
        "name": "Monitoring",
        "description": "Server health check probes.",
    },
    {
        "name": "Moteur Actuariel",
        "description": "Machine Learning inference endpoints for pricing.",
    },
]

app = FastAPI(
    title="🚗 API Tarification Auto - M2 Fintech",
    description="""
    **Production-ready REST API** for auto insurance contract pricing.
    
    ### Modeling:
    * **Frequency**: Optimal blending (Logistic Regression + XGBClassifier)
    * **Severity**: Optimal blending (Log-transformed Ridge + XGBRegressor Gamma) on the 99th percentile
    * **Pure Premium**: Strict application of the `Frequency * Amount` formula
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"syntaxHighlight.theme": "monokai"}
)

# 2. STRICT DATA CONTRACT (PYDANTIC SCHEMA)

# ContractInput class for base dataset
class ContractInput(BaseModel):
    """
    Strict schema for an insurance contract.
    Types are validated automatically. If a field is missing or invalid,
    the API will return a 422 Unprocessable Entity error.
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
    
    # The following fields can be null (None in Python)
    sex_conducteur2: Optional[str] = None
    
    anciennete_permis1: int
    anciennete_permis2: int
    
    # May be null according to coursework rules
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

# 3. API ROUTES DEFINITION

@app.get("/health", tags=["Monitoring"], summary="Check API health")
def health_check():
    """
    Liveness probe. Returns a 200 status code and status 'ok' if the server
    is operational and ready to accept connections.
    """
    return {"status": "ok"}

@app.post("/predict_frequency", tags=["Moteur Actuariel"], summary="Predict claim frequency")
def route_predict_frequency(contract: ContractInput):
    """
    Calculates claim occurrence probability for a given contract 
    via blending (LogReg + XGBClassifier).
    Returns a non-negative float.
    """
    # .model_dump() converts validated Pydantic payload into a Python dictionary
    freq = predict_frequency(contract.model_dump())
    return {"frequency": max(0.0, float(freq))}

@app.post("/predict_amount", tags=["Moteur Actuariel"], summary="Predict claim severity")
def route_predict_amount(contract: ContractInput):
    """
    Estimates claim cost if it occurs via blending 
    (Log Ridge with Duan correction + XGBRegressor Gamma).
    Returns a non-negative float.
    """
    amount = predict_amount(contract.model_dump())
    return {"amount": max(0.0, float(amount))}

@app.post("/predict", tags=["Moteur Actuariel"], summary="Calculate total pure premium")
def route_predict(contract: ContractInput):
    """
    Aggregates frequency and severity to return total pure premium 
    following strict coursework actuarial rule: premium = frequency * amount.
    """
    payload = contract.model_dump()
    
    # Call local inference module
    result = predict_premium(payload)
    
    return {
        "frequency": max(0.0, float(result["frequency"])),
        "amount": max(0.0, float(result["amount"])),
        "premium": max(0.0, float(result["premium"]))
    }

# ==============================================================================
# ENTRY POINT FOR DIRECT EXECUTION (OPTIONAL)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)