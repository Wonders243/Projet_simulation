import time
import random
import json
import math
import random
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

# predict_animal_decision.py
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle



# Chemins vers les fichiers sauvegardés
model_path = "/workspaces/Projet_simulation/simulation/model_IA/animal_decision_model1.h5"
scaler_path = "/workspaces/Projet_simulation/simulation/model_IA/scaler.pkl"
encoder_animal_path = "/workspaces/Projet_simulation/simulation/model_IA/encoder_animal.pkl"
encoder_climat_path = "/workspaces/Projet_simulation/simulation/model_IA/encoder_climat.pkl"
encoder_decision_path = "/workspaces/Projet_simulation/simulation/model_IA/encoder_decision.pkl"

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

class Simulation (AsyncWebsocketConsumer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.largeur = 1300  # Largeur du canvas en pixels
        self.hauteur = 900  # Hauteur du canvas en pixels
        self.annee = 2025
        self.mois = 3
        self.jour = 25
        self.heure = 12
        self.minute = 0
        self.seconde = 0

        self.temperature=20
        self.climat= "neige"

        self.tick_duree = 0.05  # 50 ms par tick
        self.ticks_par_jour = 72000  # 1 jour simulé = 1 heure réelle
        self.ticks_par_heure = 3000  # 1 heure simulée = 3000 ticks
        self.ticks_par_minute = 50  # 1 minute simulée = 50 ticks

        self.ticks_actuels = 0

        # Liste des animaux présents dans la simulation
        self.animaux = []

        # Création et ajout de quelques animaux dans la simulation
        self.initialiser_animaux()
    
    def initialiser_animaux(self):
        """ Initialise quelques animaux dans la simulation. """
        # Ajouter des animaux à la simulation avec leurs caractéristiques
        animaux_types = ["lapin", "ours", "lion"]
        for _ in range(10):
            type_animal = random.choice(animaux_types)
            age = random.randint(1, 10)
            poids = random.randint(10, 150)
            energie = random.randint(50, 100)
            faim = random.randint(0, 100)
            soif = random.randint(0, 100)
            self.animaux.append(Animal(type_animal, x=random.randint(0, 800), y=random.randint(0, 600), age=age, poids=poids, energie=energie, faim=faim, soif=soif))

    def animal_Action(self, test_data):

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

    def est_jour(self):
        """Détermine si c'est le jour ou la nuit."""
        return 6 <= self.heure < 18  # Jour entre 6h et 18h

    def avancer_temps(self):
        """Avance la simulation d'un tick et met à jour l'heure et la date."""
        self.ticks_actuels += 1
        self.seconde += 1

        # Gestion des secondes, minutes et heures
        if self.seconde >= 60:
            self.seconde = 0
            self.minute += 1

        if self.minute >= 60:
            self.minute = 0
            self.heure += 1

        if self.heure >= 24:
            self.heure = 0
            self.jour += 1

            # Gestion des mois et années
            if self.jour > 30:  # Supposons 30 jours par mois
                self.jour = 1
                self.mois += 1

                if self.mois > 12:  # Supposons 12 mois par an
                    self.mois = 1
                    self.annee += 1

        # Déterminer si c'est le jour ou la nuit
        cycle = "Jour" if self.est_jour() else "Nuit"

        # Affichage formaté
        print(f"{self.annee}/{self.mois}/{self.jour} - {self.heure:02}:{self.minute:02}:{self.seconde:02} [{cycle}]")

    def recuperer_donnees_animaux(self):
        """
        Cette méthode récupère toutes les données des animaux sous forme de tableau (DataFrame),
        en combinant les attributs des animaux avec ceux de la simulation.
        """
        # Liste pour stocker les données des animaux sous forme de liste
        donnees_animaux = []

        # Récupération des données globales de la simulation
        temperature = self.temperature  # Température de la simulation
        climat = self.climat  # Climat de la simulation
        heure = self.heure  # Heure actuelle de la simulation
       
        # On parcourt chaque animal et on récupère ses données spécifiques
        for animal in self.animaux:
            # On récupère les attributs de chaque animal
            nom = animal.nom
            age = animal.age
            poids = animal.poids
            energie = animal.energie
            faim = animal.faim
            soif = animal.soif
            nourriture_dispo = animal.nourriture_dispo  # Remplacer par l'attribut correct
            eau_proche = animal.eau_proche  # Remplacer par l'attribut correct
            proies = animal.proies  # Nombre de proies
            predateurs = animal.predateurs  # Nombre de prédateurs

            # Ajout des données de l'animal dans la liste sous forme de ligne
            donnees_animaux.append([
                nom, age, poids, energie, faim, soif, nourriture_dispo, eau_proche, 
                temperature, climat, predateurs, proies, heure
            ])

        # Convertir la liste de données en DataFrame pandas
        df = pd.DataFrame(donnees_animaux, columns=[
            "animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", 
            "temperature", "climat", "predateurs", "proies", "heure"
        ])

        return df

    async def connect(self):
        await self.accept()
        asyncio.create_task(self.demarrer())

    async def disconnect(self, close_code):
        print(f"Déconnexion WebSocket, code : {close_code}")
        self.running = False  # Si tu utilises une boucle, ça permet de l’arrêter proprement
    async def envoyer_donnees(self):
        """
        Envoie l'état actuel de la simulation au client WebSocket.
        """
        donnees = self.recuperer_donnees_animaux()  # Récupère les données en DataFrame
        
        # Convertir les données du DataFrame en JSON lisible
        donnees_liste = donnees.to_dict(orient="records")

        message = {
            "annee": self.annee,
            "mois": self.mois,
            "jour": self.jour,
            "heure": self.heure,
            "minute": self.minute,
            "seconde": self.seconde,
            "temperature": self.temperature,
            "climat": self.climat,
            "animaux": donnees_liste,
        }
       # await self.send(text_data=json.dumps(message))  # Envoi des données JSON au frontend

    async def demarrer(self):
       
        while True:
            self.avancer_temps()
            donnees=self.recuperer_donnees_animaux()
            actions= self.animal_Action(donnees)
            print(actions)
            await self.envoyer_donnees()
            await asyncio.sleep(self.tick_duree)  # Pause de 50ms entre chaque tick
     
    


class Animal:
    def __init__(self, nom, x, y, age, poids, energie, faim, soif, vitesse=1, vision=100, angle_vision=90):
        self.nom = nom
        self.x = x  # Position en pixels
        self.y = y
        self.age = age
        self.poids = poids
        self.energie = energie  # (0-100)
        self.faim = faim  # (0-100)
        self.soif = soif  # (0-100)
        self.nourriture_dispo = 0  # Nourriture proche
        self.eau_proche = 0  # 1 si eau proche, 0 sinon
        self.predateurs = 2
        self.proies = 1
        self.vitesse = vitesse  # Vitesse de déplacement
        self.vision = vision  # Distance de vision (rayon de détection)
        self.angle_vision = angle_vision  # Angle du champ de vision (en degrés)
        self.etat = "actif"  # L'état de l'animal (actif, fatigué, etc.)
        self.direction = 0  # Direction de l'animal (en radians)

    def distance(self, autre):
        """Calcule la distance entre cet animal et un autre."""
        return math.sqrt((self.x - autre.x) ** 2 + (self.y - autre.y) ** 2)

    def angle_entre(self, autre):
        """Calcule l'angle entre cet animal et un autre."""
        return math.degrees(math.atan2(autre.y - self.y, autre.x - self.x)) % 360

    def est_dans_vision(self, autre):
        """Vérifie si un autre animal est dans le champ de vision de cet animal."""
        # Calculer l'angle entre l'animal et l'autre
        angle = self.angle_entre(autre)
        
        # Vérifier si l'animal est dans le champ de vision (angle +/- angle_vision)
        angle_diff = abs(self.direction - angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        return angle_diff <= self.angle_vision / 2 and self.distance(autre) <= self.vision

    def voir_environ(self, autres_animaux):
        """Renvoie les animaux visibles dans un rayon et un angle définis."""
        return [autre for autre in autres_animaux if autre is not self and self.est_dans_vision(autre)]

    def se_deplacer_vers(self, cible):
        """Se rapproche d'une cible avec déplacement progressif."""
        angle = math.atan2(cible.y - self.y, cible.x - self.x)
        self.x += math.cos(angle) * self.vitesse
        self.y += math.sin(angle) * self.vitesse
        print(f"{self.nom} se déplace vers {cible.nom} [{int(self.x)}, {int(self.y)}]")

    def se_deplacer_aleatoire(self):
        """Déplacement aléatoire en pixels."""
        angle = random.uniform(0, 2 * math.pi)
        self.x = max(0, min(self.x + math.cos(angle) * self.vitesse, 800))  # 800 = Largeur max
        self.y = max(0, min(self.y + math.sin(angle) * self.vitesse, 600))  # 600 = Hauteur max
        print(f"{self.nom} se déplace aléatoirement [{int(self.x)}, {int(self.y)}]")

    def marcher_vers(self, cible):
        """Marche lentement vers la cible."""
        print(f"{self.nom} marche lentement vers {cible.nom}.")
        self.se_deplacer_vers(cible)

    def courir_vers(self, cible):
        """Court rapidement vers la cible."""
        print(f"{self.nom} court rapidement vers {cible.nom}.")
        self.vitesse *= 2  # Double la vitesse pour simuler la course
        self.se_deplacer_vers(cible)
        self.vitesse /= 2  # Restaure la vitesse normale après la course

    def fuir(self, predateur):
        """Fuit un prédateur."""
        print(f"{self.nom} fuit {predateur.nom} !")
        angle = math.atan2(self.y - predateur.y, self.x - predateur.x)
        self.x += math.cos(angle) * self.vitesse * 2  # Fuite rapide
        self.y += math.sin(angle) * self.vitesse * 2
        print(f"{self.nom} fuit en direction [{int(self.x)}, {int(self.y)}]")

    def chasser(self, proie):
        """Logique de chasse : approche, attaque, etc."""
        distance = self.distance(proie)

        if distance > 5:
            # Si la proie est trop éloignée, marche vers elle
            self.marcher_vers(proie)
        elif distance <= 5:
            # Si la proie est proche, commence l'attaque
            print(f"{self.nom} commence à attaquer {proie.nom} !")
            self.mordre(proie)

    def mordre(self, proie):
        """Effectue l'attaque sur une proie."""
        if self.energie > 10:  # Vérifie si l'animal a assez d'énergie pour mordre
            print(f"{self.nom} mord {proie.nom} !")
            proie.subir_degats(20)  # inflige des dégâts à la proie
            self.energie -= 10  # consomme de l'énergie
        else:
            print(f"{self.nom} est trop fatigué pour mordre.")

    def subir_degats(self, degats):
        """Subit des dégâts et réagit en conséquence."""
        self.poids -= degats  # Réduit la masse de l'animal (peut être lié à sa santé)
        if self.poids <= 0:
            print(f"{self.nom} est trop affaibli et meurt.")
            self.etat = "mort"  # L'animal est mort

    def fuir_ou_chasser(self, autres_animaux):
        """Décide si l'animal doit fuir ou chasser en fonction de la situation."""
        proies = self.voir_environ(autres_animaux)
        predators = [a for a in autres_animaux if isinstance(a, Animal) and self.distance(a) < 20 and a.etat == "prédateur"]

        if predators:
            # Si des prédateurs sont à proximité, l'animal doit fuir
            for predateur in predators:
                self.fuir(predateur)
        elif proies:
            # Si des proies sont à proximité, commence à chasser
            self.chasser(proies[0])

    def se_reposer(self):
        """L'animal se repose pour récupérer de l'énergie."""
        if self.energie < 100:
            print(f"{self.nom} se repose pour récupérer de l'énergie.")
            self.energie = min(100, self.energie + 10)  # Récupère 10 points d'énergie
        else:
            print(f"{self.nom} est déjà plein d'énergie.")

    def ajuster_vision(self, valeur):
        """Ajuste le champ de vision de l'animal."""
        self.vision = valeur
        print(f"{self.nom} a maintenant une vision de {self.vision} pixels.")

    def ajuster_angle_vision(self, valeur):
        """Ajuste l'angle de vision de l'animal."""
        self.angle_vision = valeur
        print(f"{self.nom} a maintenant un angle de vision de {self.angle_vision} degrés.")

    def se_reposer():
        return
    def dormir():
        return
    def chercher_de_la_nourriture():
        return
    def se_cacher():
        return
    def fuir():
        return
    def regrouper():
        return
    def explorer():
        return

#simulation = Simulation()

# Démarre la simulation manuellement
#asyncio.run(simulation.demarrer())