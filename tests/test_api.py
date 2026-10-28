# Test file for the auto pricing FastAPI API

# Import required libraries for testing
import pytest # pytest library for unit testing
from fastapi.testclient import TestClient
from app.main import app

# Initialize FastAPI test client
client = TestClient(app)

# Sample valid payload for testing
VALID_PAYLOAD = {
    "id_client": "A00000001", "id_vehicule": "V01", "id_contrat": "A00000001-V01",
    "bonus": 0.5, "type_contrat": "Maxi", "duree_contrat": 29,
    "anciennete_info": 9, "freq_paiement": "Biannual", "paiement": "No",
    "utilisation": "Retired", "code_postal": "36233", "conducteur2": "No",
    "age_conducteur1": 85, "age_conducteur2": 0,
    "sex_conducteur1": "M", "sex_conducteur2": None,
    "anciennete_permis1": 62, "anciennete_permis2": 0,
    "anciennete_vehicule": 10, "cylindre_vehicule": 1587, "din_vehicule": 98,
    "essence_vehicule": "Gasoline", "marque_vehicule": "PEUGEOT",
    "modele_vehicule": "306", "debut_vente_vehicule": 10,
    "fin_vente_vehicule": 9, "vitesse_vehicule": 182,
    "type_vehicule": "Tourism", "prix_vehicule": 20700.0, "poids_vehicule": 1210
}

# Unit tests definition

# Test health check endpoint
def test_health_check():
    """Checks that the API is online."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test frequency prediction endpoint
def test_predict_frequency():
    """Checks typing and non-negativity of frequency prediction."""
    response = client.post("/predict_frequency", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "frequency" in data
    assert isinstance(data["frequency"], float)
    assert data["frequency"] >= 0.0

# Test premium prediction endpoint
def test_predict_premium_logic():
    """Checks the strict actuarial formula required by the API contract."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    
    expected_premium = data["frequency"] * data["amount"]
    
    # Use pytest.approx to ignore floating-point rounding errors (e.g. 0.00000000001)
    assert data["premium"] == pytest.approx(expected_premium, rel=1e-5)

# Test premium prediction endpoint with invalid payload
def test_invalid_payload_returns_422():
    """
    Checks that an incomplete contract returns a 422 validation error
    instead of a 500 server error.
    """
    invalid_payload = {"id_client": "A00000001", "bonus": 0.5} # Missing fields
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422