# Utilisation d'une image Python officielle, minimale (slim)
FROM python:3.11-slim

# Désactivation du cache de bytecode et forçage de l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Création d'un utilisateur non-root pour des raisons de sécurité
RUN adduser --disabled-password --gecos '' api-user

# Définition du répertoire de travail
WORKDIR /home/api-user/projet

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et des modèles (les seuls éléments requis en prod)
COPY app/ ./app/
COPY models_export/ ./models_export/

# Attribution des droits au nouvel utilisateur
RUN chown -R api-user:api-user /home/api-user/projet
USER api-user

# Exposition du port d'écoute de l'API
EXPOSE 8000

# Commande de lancement du serveur ASGI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]