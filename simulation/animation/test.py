# predict_animal_decision.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

# Méthode pour tester le modèle
def test_animal_model(model_path, scaler_path, encoder_animal_path, encoder_climat_path, encoder_decision_path, test_data):
    # Charger le modèle
    model = keras.models.load_model(model_path)
    
    # Charger le scaler et les encodeurs
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_animal_path, 'rb') as f:
        encoder_animal = pickle.load(f)
    with open(encoder_climat_path, 'rb') as f:
        encoder_climat = pickle.load(f)
    with open(encoder_decision_path, 'rb') as f:
        encoder_decision = pickle.load(f)

    # Encodage des données de test
    test_data["animal"] = encoder_animal.transform(test_data["animal"])
    test_data["climat"] = encoder_climat.transform(test_data["climat"])

    # Normalisation des données
    test_data = scaler.transform(test_data)

    # Reshape pour LSTM
    test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

    # Prédictions
    predictions = model.predict(test_data)
    predicted_decisions = encoder_decision.inverse_transform(np.argmax(predictions, axis=1))

    return predicted_decisions

# Exemple de données à tester
test_animals = pd.DataFrame([
    ["lion", 10, 200, 30, 80, 50, 10, 1, 30, "pluie", 1, 3, 14],
    ["gazelle", 50, 80, 90, 80, 90, 30, 0, 25, "pluie", 0, 2, 8],
    ["ours", 29, 180, 50, 40, 10, 1, 0, 0, "pluie", 2, 1, 2],
    ["lapin", 10, 200, 30, 80, 50, 10, 1, 30, "pluie", 1, 3, 14],
], 

columns=["animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", "temperature", "climat", "predateurs", "proies", "heure"])

# Chemins vers les fichiers sauvegardés
model_path = "animal_decision_model.h5"
scaler_path = "scaler.pkl"
encoder_animal_path = "encoder_animal.pkl"
encoder_climat_path = "encoder_climat.pkl"
encoder_decision_path = "encoder_decision.pkl"

# Tester le modèle
predicted_decisions = test_animal_model(model_path, scaler_path, encoder_animal_path, encoder_climat_path, encoder_decision_path, test_animals)

# Affichage des résultats
for i, animal in enumerate(test_animals["animal"]):
    animal_name = test_animals.iloc[i]["animal"]  # Nom de l'animal
    print(f"{animal_name} dans ces conditions choisit de : {predicted_decisions[i]}")
    