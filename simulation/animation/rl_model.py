# train_model.py
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Définition de la fonction de décision
def determine_decision(animal, faim, soif, energie, nourriture, eau, predateurs, proies, temperature, temps):
    chasse_nuit_prob = 0.3 if animal in ["lion", "loup"] else 0.1

    if temps == "nuit":
        if energie < 30 and random.random() > chasse_nuit_prob:
            return "dormir"
        elif faim > 80 and proies > 0 and random.random() < chasse_nuit_prob:
            return "chasser"
        else:
            return "se reposer"

    if faim > 80 and proies > 0:
        return "chasser"
    elif faim > 80 and nourriture > 5:
        return "chercher de la nourriture"
    elif soif > 80 and eau == 1:
        return "boire"
    elif soif > 80 and eau == 0:
        return "chercher de l'eau"
    elif predateurs >= 2 and energie < 40:
        return "se cacher"
    elif predateurs >= 2 and energie >= 40:
        return "fuir"
    elif predateurs > 0 and energie > 50:
        return "se regrouper"
    elif temperature < -5 or temperature > 35:
        return "dormir"
    elif energie < 20:
        return "se reposer"
    elif faim < 30 and soif < 30 and energie > 60:
        return "jouer / interagir"
    else:
        return "explorer"

# Génération de données d'animaux
def generate_animal_data(n_samples=1000):
    data = []
    
    for _ in range(n_samples):
        animal = random.choice(["lion", "gazelle", "loup", "lapin", "ours"])
        age = random.randint(0, 15)
        poids = random.randint(5, 250)
        energie = random.randint(0, 100)
        faim = random.randint(0, 100)
        soif = random.randint(0, 100)
        nourriture = random.randint(0, 50)
        eau = random.choice([0, 1])
        temperature = random.randint(-10, 40)
        climat = random.choice(["pluie", "soleil", "neige", "vent"])
        predateurs = random.randint(0, 2)
        proies = random.randint(0, 5)
        temps = random.randint(0, 23)
        
        decision = determine_decision(animal, faim, soif, energie, nourriture, eau, predateurs, proies, temperature, temps)
        data.append([animal, age, poids, energie, faim, soif, nourriture, eau, temperature, climat, predateurs, proies, temps, decision])
    
    columns = ["animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", "temperature", "climat", "predateurs", "proies", "heure", "decision"]
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Génération des données
df = generate_animal_data(100000)

# Encodage des variables
encoder_animal = LabelEncoder()
df["animal"] = encoder_animal.fit_transform(df["animal"])

encoder_climat = LabelEncoder()
df["climat"] = encoder_climat.fit_transform(df["climat"])

encoder_decision = LabelEncoder()
df["decision"] = encoder_decision.fit_transform(df["decision"])

# Séparation des features et labels
X = df.drop(columns=["decision"])
y = df["decision"]

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape pour LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Définition du modèle LSTM
model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True, input_shape=(1, X.shape[2])),
    keras.layers.LSTM(50),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(len(df["decision"].unique()), activation='softmax')
])

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Sauvegarde du modèle et des encodeurs
model.save("animal_decision_model1.h5")
scaler_filename = "scaler.pkl"
encoder_animal_filename = "encoder_animal.pkl"
encoder_climat_filename = "encoder_climat.pkl"
encoder_decision_filename = "encoder_decision.pkl"

# Sauvegarder les encodeurs
import pickle
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
with open(encoder_animal_filename, 'wb') as f:
    pickle.dump(encoder_animal, f)
with open(encoder_climat_filename, 'wb') as f:
    pickle.dump(encoder_climat, f)
with open(encoder_decision_filename, 'wb') as f:
    pickle.dump(encoder_decision, f)

print("Modèle et encodeurs sauvegardés.")
