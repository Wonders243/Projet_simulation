# train_model.py
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def determine_decision(animal, faim, energie, nourriture, predateurs, pret_reproduction, partenaire_proche):
    """
    Version simplifiée avec 5 actions principales :
    1. fuir (si prédateurs)
    2. chasser (pour les prédateurs si faim)
    3. manger (si nourriture disponible)
    4. se reproduire (si conditions remplies)
    5. explorer (par défaut)
    """
    # 1. Danger immédiat
    if energie < 3:
        return "mourir"
    
    # 1. Danger immédiat
    if predateurs > 0:
        return "fuir"
    
    # 2. Chasser (pour les prédateurs)
    if faim > 50 and animal in ["lion", "ours"] and energie < 30:
        return "chasser"
    
    # 3. Manger/boire
    if (faim > 40 and nourriture == 1):
        return "manger"
    
    # 4. Reproduction
    if pret_reproduction==1 and partenaire_proche ==1 and energie > 40 and faim < 60:
        return "se reproduire"
    
    # 5. Action par défaut
    return "explorer"

# Génération de données d'animaux simplifiée
def generate_animal_data(n_samples=50000):
    data = []
    animals = ["lion","lapin", "ours"]
    
    for _ in range(n_samples):
        animal = random.choice(animals)
        energie = random.randint(0, 100)
        faim = random.randint(0, 100)
        soif = random.randint(0, 100)
        nourriture = random.choice([0, 1])
        eau = random.choice([0, 1])
        predateurs = random.choice([0, 1, 2])  # 0 = pas de prédateur
        heure = random.randint(0, 23)
        pret_reproduction = random.choice([0, 1])
        partenaire_proche = random.choice([0, 1])

        decision = determine_decision(
            animal, faim, energie, nourriture,
            predateurs,
            pret_reproduction, partenaire_proche
        )

        data.append([
            animal, energie, faim, nourriture, predateurs, 
            pret_reproduction, partenaire_proche, decision
        ])

    columns = [
        "animal", "energie", "faim", "nourriture",
        "predateurs", "pret_reproduction", "partenaire_proche", "decision"
    ]

    df = pd.DataFrame(data, columns=columns)
    return df

# Génération des données
df = generate_animal_data(30000)

# Encodage des variables
encoder_animal = LabelEncoder()
df["animal"] = encoder_animal.fit_transform(df["animal"])

encoder_decision = LabelEncoder()
df["decision"] = encoder_decision.fit_transform(df["decision"])

# Séparation des features et labels
X = df.drop(columns=["decision"])
y = df["decision"]

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Application de SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)


# Modèle simplifié (Dense au lieu de LSTM pour plus de rapidité)
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_res.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(np.unique(y_res)), activation='softmax')
])

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_res, y_res, epochs=10, batch_size=32, validation_split=0.2)

# Sauvegarde du modèle et des encodeurs
model.save("/workspaces/Projet_simulation/simulation/model_IA/animal_decision_model1.h5")
scaler_filename = "/workspaces/Projet_simulation/simulation/model_IA/scaler.pkl"
encoder_animal_filename = "/workspaces/Projet_simulation/simulation/model_IA/encoder_animal.pkl"
encoder_decision_filename = "/workspaces/Projet_simulation/simulation/model_IA/encoder_decision.pkl"

# Sauvegarder les encodeurs
import pickle
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
with open(encoder_animal_filename, 'wb') as f:
    pickle.dump(encoder_animal, f)
with open(encoder_decision_filename, 'wb') as f:
    pickle.dump(encoder_decision, f)

# Sauvegarde des données d'entraînement
df.to_csv("/workspaces/Projet_simulation/simulation/model_IA/donnees_entrainement.csv", index=False)
print("Données d'entraînement sauvegardées.")
print("Modèle et encodeurs sauvegardés.")