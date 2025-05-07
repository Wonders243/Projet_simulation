# train_model.py
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE  # Import de SMOTE

import random
import random

import random

def determine_decision(animal, faim, soif, energie, nourriture, eau, predateurs, temps, pret_reproduction, partenaire_proche):
    """
    Version simplifiée avec 6 actions principales :
    1. mourir (si énergie <= 0)
    2. fuir (si prédateurs)
    3. satisfaire besoins (boire/manger)
    4. se reposer (si fatigue)
    5. se reproduire (si conditions remplies)
    6. explorer (par défaut)
    """
    
    # 1. Mort impérative
    if energie < 2.00:
        return "mourir"
    
    # 2. Danger immédiat (priorité absolue)
    if predateurs > 0:
        return "se cacher" if energie < 30.00 and predateurs > 3 else "fuir"
    
    # 4. Besoins physiologiques (fusion boire/manger)
    
    if faim > 75.00 or (faim > 50.00 and random.random() < 0.6):
       
        if nourriture == 1:
            return "satisfaire besoin"
        elif animal in ["lion", "ours"]:
            return "chasser"
        else:
            return "chercher ressources"
        
    if soif > 80.00 or (soif > 60.00 and random.random() < 0.7):
        return "satisfaire besoin" if eau or nourriture ==1 else "chercher ressources"
    
    # 5. Fatigue
    if energie < 30 or (temps < 6 and temps > 6 and energie < 70):
        return "se reposer"
    
    # 6. Reproduction
    if pret_reproduction and energie > 40 and faim < 60 and soif < 60:
        return "se reproduire" if partenaire_proche else "chercher partenaire"
    
    # 7. Action par défaut
    return "explorer"

# Génération de données d'animaux
def generate_animal_data(n_samples=80000):
    data = []

    # Définition des plages par niveau
    niveau = {
        "energie": [random.randint(0, 4), random.randint(10, 15), random.randint(21, 40), random.randint(41, 60), random.randint(75, 100)],
        "faim": [random.randint(0, 15), random.randint(26, 50), random.randint(51, 75), random.randint(76, 100)],
        "soif": [random.randint(0, 25), random.randint(51, 75), random.randint(76, 100)],
        "temperature": [random.randint(-15, 0), random.randint(1, 15), random.randint(31, 45)],
        "age": [random.randint(0, 3),  random.randint(8, 11), random.randint(12, 15)],
        "poids": [random.randint(5, 50), random.randint(101, 175), random.randint(176, 250)],
        "predateurs": [0, 1, 5],  
        "heure": [0, 6, 18]
    }

    for _ in range(n_samples):
        animal = random.choice(["lion", "gazelle", "loup", "lapin", "ours"])
        age = random.choice(niveau["age"])
        poids = random.choice(niveau["poids"])
        energie = random.choice(niveau["energie"])
        faim = random.choice(niveau["faim"])
        soif = random.choice(niveau["soif"])
        nourriture = random.choice([0, 1])
        eau = random.choice([0, 1])
       
        predateurs = random.choice(niveau["predateurs"])
        
        heure = random.choice(niveau["heure"])
        partenaire_proche = random.choice([0, 1])
        pret_reproduction = random.choice([0, 1])

        decision = determine_decision(
            animal, faim, soif, energie, nourriture, eau,
            predateurs, heure,
            pret_reproduction, partenaire_proche
        )

        data.append([
            animal, age, poids, energie, faim, soif, nourriture, eau, predateurs, heure,
            pret_reproduction, partenaire_proche, decision
        ])

    columns = [
        "animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau",
        "predateurs",  "heure", "pret_reproduction", "partenaire_proche", "decision"
    ]

    df = pd.DataFrame(data, columns=columns)
    return df

# Génération des données
df = generate_animal_data(100000)

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

# Application de SMOTE pour équilibrer les classes
smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' pour sur-échantillonner les classes minoritaires
X_res, y_res = smote.fit_resample(X, y)

from collections import Counter
print("Distribution AVANT SMOTE :", Counter(y))
print("Distribution APRÈS  SMOTE :", Counter(y_res))
# Reshape pour LSTM
X_res = X_res.reshape((X_res.shape[0], 1, X_res.shape[1]))

# Définition du modèle LSTM
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(1, X_res.shape[2])),
    keras.layers.LSTM(50),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(len(df["decision"].unique()), activation='softmax')
])

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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