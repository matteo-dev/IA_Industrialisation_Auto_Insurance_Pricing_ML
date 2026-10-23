# Fichier de test pour l'API FastAPI de tarification auto

# Importation des bibliothèques nécessaires pour les tests
import pytest # Bibliothèque pytest pour les tests unitaires
from fastapi.testclient import TestClient
from app.main import app

# Initialisation du client de test FastAPI
client = TestClient(app)

# Payload d'exemple valide pour les tests
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

# Définition des tests unitaires

# Test de la route de santé
def test_health_check():
    """Vérifie que l'API est en ligne."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test de la route de prédiction de prime
def test_predict_frequency():
    """Vérifie le typage et la positivité de la prédiction de fréquence."""
    response = client.post("/predict_frequency", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "frequency" in data
    assert isinstance(data["frequency"], float)
    assert data["frequency"] >= 0.0

# Test de la route de prédiction de montant
def test_predict_premium_logic():
    """Vérifie la formule actuarielle stricte imposée par le contrat d'API."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    
    expected_premium = data["frequency"] * data["amount"]
    
    # Utilisation de pytest.approx pour ignorer les micro-variations flottantes (ex: 0.00000000001)
    assert data["premium"] == pytest.approx(expected_premium, rel=1e-5)

# Test de la route de prédiction de prime avec un payload invalide
def test_invalid_payload_returns_422():
    """
    Vérifie qu'un contrat incomplet renvoie une erreur de validation 422 
    et non une erreur serveur 500.
    """
    
    invalid_payload = {"id_client": "A00000001", "bonus": 0.5} # Champs manquants
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422