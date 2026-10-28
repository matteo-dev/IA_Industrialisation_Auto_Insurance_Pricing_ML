# API de Tarification d'Assurance Auto - Industrialisation ML

*(English version below)*

Ce dépôt contient le projet complet d'industrialisation algorithmique (MLOps) réalisé dans le cadre du cours d'Industrialisation IA du Master 2 Fintech de CY Tech. Il vise à transformer un script monolithique d'entraînement de modèles (Machine Learning & Actuariat) en une API REST robuste, testée et prête pour la production.

## Stack Technique & Infrastructure
* **Langage & Environnement :** Python 3.12, `uv` (gestionnaire ultra-rapide de paquets et d'environnements)
* **Framework API :** FastAPI, Uvicorn, Pydantic (validation stricte des schémas de données)
* **Machine Learning :** Scikit-Learn, XGBoost, Optuna, Joblib
* **Qualité & Tests :** Pytest, HTTPX (TestClient)
* **DevOps & CI/CD :** Docker, GitHub Actions (CI pipeline), Render (Déploiement Continu)

## Architecture du Projet
Le projet sépare strictement la phase de recherche/entraînement de l'environnement d'inférence en production :

* `app/` : Code source de l'API en production.
  * `main.py` : Application FastAPI, définition des routes et schémas Pydantic.
  * `inference.py` : Chargement des modèles en mémoire (Singleton) et logique de prédiction.
  * `processing.py` : Feature engineering unitaire et déterministe pour les payloads JSON.
* `models_export/` : Artefacts des modèles sérialisés (`.joblib`). Contient les préprocesseurs, les modèles Ridge/XGBoost et les poids de blending. *(Note : exceptionnellement versionnés sur Git pour les besoins académiques du déploiement cloud).*
* `tests/` : Suite de tests automatisés (`pytest`) validant la cohérence mathématique, le contrat d'API et la gestion des erreurs HTTP 422.
* `training/` : Scripts de recherche originaux en Data Science (`step_final.py`). Intègre la logique actuarielle avancée (Capping, Blending, correction de Duan).
* `.github/workflows/` : Configuration de l'Intégration Continue (CI) pour installer les dépendances, exécuter les tests et tester le build Docker à chaque `push`.

## Logique Actuarielle & Machine Learning
L'API calcule la **prime pure** d'un contrat d'assurance automobile de façon unitaire, en respectant scrupuleusement la formule académique : `premium = frequency * amount`.

* **Modèle de Fréquence :** Blending optimisé entre une Régression Logistique bien calibrée et un XGBClassifier (optimisé via Optuna).
* **Modèle de Gravité (Montant) :** Blending entre une Régression Ridge log-transformée (avec correction de l'estimateur de Duan) et un XGBRegressor (fonction de perte Gamma). Entraîné exclusivement sur des données écrêtées au 99e percentile pour isoler le risque attritionnel.
* **Feature Engineering :** Les variables dynamiques (ex: extraction du département, ratio puissance/poids, calcul de l'ancienneté de permis) sont encapsulées directement dans le pipeline de traitement de l'API.

## Contrat d'API (Endpoints)
L'API impose une validation stricte des entrées. Des champs manquants ou mal typés (ex: une chaîne de caractères à la place d'un entier) déclencheront systématiquement une erreur `HTTP 422 Unprocessable Entity`, évitant ainsi les crashs serveurs (500).

| Endpoint | Méthode | Entrée | Exemple de Sortie |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Aucune | `{"status": "ok"}` |
| `/predict_frequency` | `POST` | Contrat JSON | `{"frequency": 0.12}` |
| `/predict_amount` | `POST` | Contrat JSON | `{"amount": 1500.0}` |
| `/predict` | `POST` | Contrat JSON | `{"frequency": 0.12, "amount": 1500.0, "premium": 180.0}` |

*La documentation interactive est accessible nativement sur `/docs` (Swagger UI) et `/openapi.json`.*

## Déploiement Local

**1. Cloner le dépôt et initialiser l'environnement**
```bash
git clone https://github.com/matteo-dev/IA_Industrialisation_Auto_Insurance_Pricing_ML.git
cd IA_Industrialisation_Auto_Insurance_Pricing_ML
uv sync
```

**2. Lancer localement avec Uvicorn (Mode Développement)**
```bash
uv run uvicorn app.main:app --reload
```
L'interface Swagger sera disponible sur : `http://127.0.0.1:8000/docs`

**3. Exécuter la suite de tests automatisés**
```bash
uv run pytest tests/
```

**4. Construire et lancer via Docker (Mode Production)**
```bash
docker build -t api-tarification-auto:latest .
docker run -p 8000:8000 api-tarification-auto:latest
```

## Déploiement Cloud (CI/CD)
Ce projet est configuré pour un déploiement continu sur **Render**. Chaque `push` sur la branche `main` déclenche une GitHub Action qui valide les tests unitaires. En cas de succès, Render récupère le dernier commit, construit l'image Docker et la déploie sur une URL publique "Live".

---
---

# Auto Insurance Pricing API - MLOps Pipeline

This repository contains the end-to-end algorithmic industrialization (MLOps) project developed for the **AI Industrialization course (M2 Fintech, CY Tech)**. The objective is to transform a monolithic machine learning and actuarial training script into a robust, fully tested, and production-ready REST API.

## Tech Stack & Infrastructure
* **Language & Environment:** Python 3.12, `uv` (Fast Python package and project manager)
* **API Framework:** FastAPI, Uvicorn, Pydantic (Strict schema validation)
* **Machine Learning:** Scikit-Learn, XGBoost, Optuna, Joblib
* **Quality & Testing:** Pytest, HTTPX (TestClient)
* **DevOps & CI/CD:** Docker, GitHub Actions (CI pipeline), Render (CD deployment)

## Project Architecture
The project strictly separates the research/training phase from the production inference environment:

* `app/`: Production API source code.
  * `main.py`: FastAPI application, route definitions, and Pydantic schemas.
  * `inference.py`: Singleton model loading and prediction logic.
  * `processing.py`: Unitary, deterministic feature engineering for incoming JSON payloads.
* `models_export/`: Serialized model artifacts (`.joblib`). Contains fitted preprocessors, Ridge/XGBoost models, and blending weights. *(Note: Tracked via Git for academic deployment purposes).*
* `tests/`: Automated test suite (`pytest`) validating mathematical coherence, API contracts, and HTTP 422 error handling.
* `training/`: Original data science research scripts (`step_final.py`). Features advanced Actuarial/ML logic (capping, blending, Duan's correction).
* `.github/workflows/`: Continuous Integration (CI) configuration to automatically install dependencies, run tests, and build the Docker image on every push.

## Actuarial & Machine Learning Logic
The API calculates the **pure premium** of an auto insurance contract on a unitary basis, strictly following the academic contract: `premium = frequency * amount`.

* **Frequency Model:** Optimized blending between a well-calibrated Logistic Regression and an XGBClassifier (optimized via Optuna).
* **Severity Model (Amount):** Blending between a Log-transformed Ridge Regression (with Duan's smearing estimator correction) and an XGBRegressor (Gamma loss function). Trained exclusively on the 99th percentile capped data to capture attrition risk.
* **Feature Engineering:** Dynamic features (e.g., zip code parsing, power-to-weight ratios, driving license age calculation) are encapsulated directly within the API's processing pipeline.

## API Contract (Endpoints)
The API strictly enforces input validation. Missing or wrongly typed fields (e.g., passing a string instead of an integer) will systematically trigger an `HTTP 422 Unprocessable Entity` error, preventing internal server errors (500).

| Endpoint | Method | Input | Output Example |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `{"status": "ok"}` |
| `/predict_frequency` | `POST` | Contract JSON | `{"frequency": 0.12}` |
| `/predict_amount` | `POST` | Contract JSON | `{"amount": 1500.0}` |
| `/predict` | `POST` | Contract JSON | `{"frequency": 0.12, "amount": 1500.0, "premium": 180.0}` |

*Interactive documentation is available natively at `/docs` (Swagger UI) and `/openapi.json`.*

## Local Deployment

**1. Clone the repository and initialize the environment**
```bash
git clone https://github.com/matteo-dev/IA_Industrialisation_Auto_Insurance_Pricing_ML.git
cd IA_Industrialisation_Auto_Insurance_Pricing_ML
uv sync
```

**2. Run locally with Uvicorn (Development mode)**
```bash
uv run uvicorn app.main:app --reload
```
Access the Swagger UI at: `http://127.0.0.1:8000/docs`

**3. Run the automated test suite**
```bash
uv run pytest tests/
```

**4. Build and run with Docker (Production mode)**
```bash
docker build -t api-tarification-auto:latest .
docker run -p 8000:8000 api-tarification-auto:latest
```

## Cloud Deployment (CI/CD)
This project is configured for continuous deployment on **Render**. Every push to the `main` branch triggers a GitHub Action that runs the unit tests. If successful, Render automatically pulls the latest commit, builds the Docker image, and deploys it on a live URL.