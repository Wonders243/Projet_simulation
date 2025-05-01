
import random
import time
import json
import math
import random
from collections import deque
import asyncio
from channels.layers import get_channel_layer
from channels.generic.websocket import AsyncWebsocketConsumer

# predict_animal_decision.py
import pandas as pd
import numpy as np
from tensorflow import keras
import pickle
import os
import platform

def clear_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')



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
        self.channel_layer = get_channel_layer()
        self.largeur = 10000  # Largeur du canvas en pixels
        self.hauteur = 10000  # Hauteur du canvas en pixels
        self.territoires = {
            'lion': (9203, 963),
            'lapin': (124, 9004),
            'gazelles':(1783, 3412),
            'ours': (8074, 11023)
        }
        
        self.annee = 2025
        self.mois = 3
        self.jour = 25
        self.heure = 21
        self.minute = 0
        self.seconde = 0

        self.temperature=20
        self.climat= "neige"

        self.tick_duree = 0.01  # 10 ms par tick
        self.ticks_par_heure = 2000  # 1 heure simulée = 20 secondes réelles
        self.ticks_par_minute = 33  # 1 minute simulée = 33 ticks
        self.ticks_par_jour = 48000  # 1 jour simulé = 48000 ticks = 8 minutes réelles
        self.ticks_actuels =0
        # Liste des animaux présents dans la simulation
        self.animaux = []

        self.ressources = self.generer_ressources()

        # Création et ajout de quelques animaux dans la simulation
        self.initialiser_animaux()
    
    def initialiser_animaux(self):
        """ Initialise quelques animaux dans la simulation. """
        # Ajouter des animaux à la simulation avec leurs caractéristiques
        animaux_types = ["lapin", "lion", "ours"]
        for _ in range(51):
            type_animal = random.choice(animaux_types)
            age = random.randint(1, 10)
            poids = random.randint(10, 300)
            energie = random.randint(24, 100)
            faim = random.randint(0, 100)
            soif = random.randint(0, 80)
           
            if type_animal == "lion":
                territoire = self.territoires['lion']
            elif type_animal == "lapin":
                territoire = self.territoires['lapin']
            elif type_animal == "ours":
                territoire = self.territoires['ours']
            else:
                territoire = {"x": 1000, "y": 1000}  # Par défaut, si un type inconnu (par sécurité)



            self.animaux.append(Animal(type_animal, x=random.randint(0, 2000), y=random.randint(0, 2000), age=age, poids=poids, energie=energie, faim=faim, soif=soif, territoire=territoire))

    def generer_ressources(self, test=100, n_zones_eau=10):
        ressources = []
        for _ in range(n_zones_eau):
            x = random.randint(0, 10000)
            y = random.randint(0, 10000)
            rayon = random.randint(300, 800)
            ressources.append(Eau(x, y, rayon))

        for _ in range(test):
            x = random.randint(0, 1000)
            y = random.randint(0, 1000)
            ressources.append(Plante(x, y))
        
       

        return ressources

        


    def animal_Action(self, test_data):
        # Vérifie que test_data est bien un DataFrame
        if not isinstance(test_data, pd.DataFrame):
            test_data = pd.DataFrame(test_data)

        # Remet les colonnes dans l'ordre attendu par le scaler
        colonnes = list(scaler.feature_names_in_)
        test_data = test_data[colonnes]

        # Encodage des colonnes catégorielles
        test_data["animal"] = encoder_animal.transform(test_data["animal"])
        test_data["climat"] = encoder_climat.transform(test_data["climat"])
    
        # Normalisation
        test_data = scaler.transform(test_data)

        # Reshape pour LSTM
        test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

        
        # Prédictions
        predictions = model.predict(test_data)
        predicted_decisions =encoder_decision.inverse_transform(np.argmax(predictions, axis=1))

        return predicted_decisions

    def est_jour(self):
        """Détermine si c'est le jour ou la nuit."""
        return 6 <= self.heure < 18  # Jour entre 6h et 18h

    def avancer_temps(self):
        self.ticks_actuels += 1

        # Conversion directe du nombre de ticks en temps simulé
        total_seconds = self.ticks_actuels

        self.seconde = total_seconds % 60
        self.minute = (total_seconds // 60) % 60
        self.heure = (total_seconds // 3600) % 24
        self.jour = (total_seconds // (3600 * 24)) % 30 + 1
        self.mois = (total_seconds // (3600 * 24 * 30)) % 12 + 1
        self.annee = 2025 + (total_seconds // (3600 * 24 * 365))

        self.cycle = "Jour" if self.est_jour() else "Nuit"

    def recuperer_donnees_animaux_transferer(self):
        """
        Récupère deux DataFrames :
        - Un tableau des animaux avec leurs caractéristiques et contexte environnemental.
        - Un tableau global des ressources (plantes et eau) avec position et type.
        
        Args:
            ressources (list): Liste d'objets Plante ou Eau (avec attributs .x, .y et .type)
            
        Returns:
            tuple: (df_animaux, df_ressources)
        """
        donnees_animaux = []
        donnees_ressources = []
        points_deau =[]

        # Données globales
        temperature = self.temperature 
        climat = self.climat 
        heure = self.heure 

        # ----- 1. Données des animaux -----
        for animal in self.animaux:
            donnees_animaux.append([
                animal.nom,
                animal.age,
                animal.poids,
                animal.energie,
                animal.faim,
                animal.soif,
                animal.nourriture_dispo,
                animal.eau_proche,
                temperature,
                climat,
                animal.predateurs,
                animal.proies,
                heure,
                animal.x,
                animal.y,
                animal.color
            ])

        df_animaux = pd.DataFrame(donnees_animaux, columns=[
            "animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", 
            "temperature", "climat", "predateurs", "proies", "heure", "x", "y", "color"
        ])

        # ----- 2. Données des ressources -----
        for res in self.ressources:
            donnees_ressources.append([
                res.type,
                res.x,
                res.y,
            ])
            if (res.type== "eau"):
                for point in res.points:
                    points_deau.append(point)
                


       
        df_points = pd.DataFrame(points_deau, columns=["x", "y"])
        df_ressources = pd.DataFrame(donnees_ressources, columns=["type", "x", "y"])

        return df_animaux, df_ressources, df_points


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
            partenaire_proche=animal.partenaire_proche
            pret_reproduction=animal.pret_reproduction

            # Ajout des données de l'animal dans la liste sous forme de ligne
            donnees_animaux.append([
                nom, age, poids, energie, faim, soif, nourriture_dispo, eau_proche, 
                temperature, climat, predateurs, proies, heure, pret_reproduction, partenaire_proche
            ])

        # Convertir la liste de données en DataFrame pandas
        df = pd.DataFrame(donnees_animaux, columns = ["animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", "temperature", "climat", "predateurs", "proies", "heure",  "pret_reproduction", "partenaire_proche"])
       
        return df
    
    async def connect(self):
            # Lors de la connexion, accepter la connexion WebSocket
            self.room_name = "animals"
            self.room_group_name = f"ws_{self.room_name}"

            # Rejoindre un groupe WebSocket
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )

            # Accepter la connexion WebSocket
            await self.accept()
            asyncio.create_task(self.demarrer())


    async def disconnect(self, close_code):
        # Quitter le groupe WebSocket
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def envoyer_donnees(self):
        """
        Envoie l'état actuel de la simulation au client WebSocket.
        """
        # Récupère les données des animaux et des ressources sous forme de DataFrames
        df_animaux, df_ressources, df_points = self.recuperer_donnees_animaux_transferer()
        
        # Convertir les données des DataFrames en JSON lisible
        donnees_animaux = df_animaux.to_dict(orient="records")
        donnees_ressources = df_ressources.to_dict(orient="records")
        donnees_eau=df_points.to_dict(orient="records")
        # Créer le message à envoyer
        message = {
            "annee": self.annee,
            "mois": self.mois,
            "jour": self.jour,
            "heure": self.heure,
            "minute": self.minute,
            "seconde": self.seconde,
            "temperature": self.temperature,
            "climat": self.climat,
            "animaux": donnees_animaux,
            "ressources": donnees_ressources,  # Ajoute les ressources dans le message
            "points": donnees_eau
        }
        
        # Envoi des données JSON au frontend via WebSocket
        await self.send(text_data=json.dumps(message))

    def interpreter(self, animal, action):
        """Interprète l'action et appelle la méthode correspondante sur l'animal.
        
        Args:
            animal: L'instance de l'animal
            action: L'action à exécuter (str)
        """
        # Dictionnaire complet des actions avec leurs méthodes correspondantes
        actions_disponibles = {
            # Actions vitales
            "mourir": animal.mourir,
            
            # Réponses aux dangers
            "fuir": animal.fuir,
            "se cacher": animal.se_cacher,
            "rester vigilant": animal.rester_alerte,
            "chercher un abri": animal.chercher_abri,
            
            # Besoins physiologiques
            "boire": animal.boire,
            "chercher de l'eau": lambda: animal.chercher_ressource('eau'),
            "manger": animal.manger,
            "chasser": self._executer_chasse,
            "chercher de la nourriture": lambda: animal.chercher_ressource('nourriture'),
            
            # Reproduction
            "se reproduire": animal.se_reproduire,
            "chercher un partenaire": animal.chercher_partenaire,
            
            # Comportements
            "se reposer": animal.se_reposer,
            "dormir": animal.dormir,
            "explorer": animal.explorer,
            "jouer / interagir": animal.jouer_interagir,
            
            # Actions composites
            "satisfaire besoin": self._satisfaire_besoin,
            "chercher ressources": animal.explorer_librement_ressources
        }
        
        # Exécution de l'action
        action_finale = animal.determiner_action_finale(action)
        
        if action_finale in actions_disponibles:
            if action_finale == "chasser":
                actions_disponibles[action_finale](animal)
            elif action_finale in ["satisfaire besoin"]:
                actions_disponibles[action_finale](animal)
            else:
                actions_disponibles[action_finale]()
        else:
            animal.explorer()  # Action par défaut

    def _executer_chasse(self, animal):
        """Gère la logique de chasse"""
        if hasattr(animal, 'listproies') and len(animal.listproies) > 0:
            proie_proche = min(animal.listproies, key=lambda a: animal.distance(a))
            animal.chasser(proie_proche)
        else:
            animal.chercher_ressource('nourriture')

    def _satisfaire_besoin(self, animal):
        """Gère la satisfaction des besoins (eau ou nourriture)"""
        if animal.soif > animal.faim:
            if animal.environnement['eau_disponible']:
                animal.boire()
            else:
                animal.chercher_ressource('eau')
        else:
            if animal.environnement['nourriture_disponible']:
                animal.manger()
            else:
                animal.chercher_ressource('nourriture')


    async def demarrer(self):
        while True:
            #clear_console()
            self.avancer_temps()
            donnees=self.recuperer_donnees_animaux()
            actions= self.animal_Action(donnees)
            print(donnees)
            print(actions)
            for i, animal in enumerate(self.animaux):
                if(animal.est_vivant==True):
                    animal.mise_a_jour(self.animaux, self.ressources)
                    action = str(actions[i]).lower()
                    self.interpreter(animal, action) 
                   

            await self.envoyer_donnees()
            await asyncio.sleep(self.tick_duree) 


class Animal:
    def __init__(self, nom, x, y, age, poids, energie, faim, soif,  territoire, vitesse=3, vision=100, rayon_territoire=100):
        self.nom = nom
        self.color= self.attribuer_couleur()
        self.vitesse = self.attribuer_vitesse()
        self.vitesse_max = self.attribuer_vitesse_max()
        self.vision = self.attribuer_vision()
        self.angle_vision = self.attribuer_angle_vision()
        self.acceleration = self.attribuer_acceleration()
        

        self.x = x 
        self.y = y
        self.dx =1
        self.dy=1
        self.age = age
        self.poids = poids
        self.energie =energie
        self.vitesse_possible=self.vitesse_max
        self.faim = faim  
        self.soif = soif  
        self.nourriture_dispo = 0  
        self.eau_proche = 1  
        self.predateurs = 0
        self.proies = 0
        self.listpredateurs = []
        self.listproies = []
        self.plantes = [] 
        self.eaux = []
        self.vision = 100  
       
        self.etat = "actif" 
        self.partenaire_proche=random.choice([0, 1])
        self.pret_reproduction=random.choice([0, 1])
        self.angle = 0  

        self.fatigue=0
        self.nourriture_mangée=0
        self.vitesse=vitesse
        self.territoire_x, self.territoire_y = territoire
        self.rayon_territoire = rayon_territoire
        self.memoire_actions =  deque(maxlen=3) 
        self.memoire_zones = deque(maxlen=50) 
        self.est_vivant = True

    def mourir(self):
        self.etat = "mort"
        self.est_vivant = False
        print(f"{self.nom} est mort.")

    def memoriser_action(self, action):
        """Ajoute une action à la mémoire de l'animal (FIFO de 3 éléments)."""
        self.memoire_actions.append(action)
        
    def est_action_prioritaire(self, action):
        """Détermine si l'action donnée est prioritaire en fonction de la vitesse, de la fatigue, etc."""
        
        # Cas où fuir ou se cacher a une priorité absolue
        if action == "fuir" or action == "se cacher":
            return True
        
        # Cas où chasser n'est pas prioritaire si l'animal est fatigué ou en pleine course
        if action == "chasser" and (self.fatigue > 70 or self.vitesse > self.vitesse_possible * 0.7):
            return False  # Si l'animal est déjà trop fatigué ou rapide, la chasse devient moins prioritaire

        # Si l'animal est fatigué, dormir ou se reposer peut devenir prioritaire pour récupérer
        if action == "dormir" or action == "se reposer":
            if self.fatigue > 50:  # Si l'animal est très fatigué, ces actions deviennent importantes
                return True
            else:
                return False
        
        # Sinon, aucune action n'est prioritaire par rapport à une autre
        return False

    def determiner_action_finale(self, action_importante):
        # Si on détecte un schéma A-B-A avec une action B non prioritaire
        if (
            len(self.memoire_actions) == 3
            and self.memoire_actions[0] == self.memoire_actions[2]
            and not self.est_action_prioritaire(action_importante)
            and self.memoire_actions[1] == action_importante
        ):  
            
            return self.memoire_actions[2]

        self.memoriser_action(action_importante)
        return action_importante
    
    def vitesse_max_atteignable(self):
        """
        Calcule la vitesse maximale que peut atteindre l'animal selon son état.
        La fatigue a un effet très marqué à partir de 70%.
        """
        vmax = self.vitesse_max

        # Réduction non linéaire liée à la fatigue
        if self.fatigue <= 70:
            reduction_fatigue = self.fatigue / 100  # normale en-dessous de 70%
        else:
            # Fatigue sévère : on accentue la pénalité exponentiellement
            reduction_fatigue = 0.7 + ((self.fatigue - 70) / 30) ** 2  # ex : à 80% → ~1.11 ; à 100% → ~1.77
            reduction_fatigue = min(reduction_fatigue, 2)  # clamp pour éviter que ce soit trop extrême

        # Autres facteurs plus doux
        reduction_age = self.age / 100
        reduction_poids = self.poids / 500
        reduction_energie = (100 - self.energie) / 100

        # Poids des facteurs dans la réduction totale
        coefficient_total = 1 - (
            reduction_fatigue +
            0.002 * reduction_age +
            0.002 * reduction_poids +
            0.002 * reduction_energie
        )

        # Clamp du coefficient pour éviter des vitesses trop faibles
        coefficient_total = max(0.1, min(1, coefficient_total))

        self.vitesse_possible = vmax * coefficient_total
        
    def distance(self, autre):
        """Calcule la distance entre cet animal et un autre. pytagore"""
        return math.sqrt((self.x - autre.x) ** 2 + (self.y - autre.y) ** 2)

    def angle_entre(self, autre):
        """Calcule l'angle entre cet animal et un autre."""
        return math.degrees(math.atan2(autre.y - self.y, autre.x - self.x)) % 360

    def est_dans_vision(self, autre):
        """Vérifie si un autre animal est dans le champ de vision de cet animal."""
        angle = self.angle_entre(autre)
        angle_diff = abs(self.angle - angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        return angle_diff <= self.angle_vision / 2 and self.distance(autre) <= self.vision or self.distance(autre) <= 10

    def mettre_a_jour_environnement(self, autres_animaux, Ressources):
        """
        Met à jour les listes de prédateurs, proies, plantes et zones d'eau dans le champ de vision.

        Args:
            autres_animaux (list): Liste de tous les autres animaux dans l'environnement.
            ressources (list): Liste de toutes les ressources (plantes et eau).
        """
        # Réinitialisation des ensembles pour éviter les doublons
        predateurs_temporaires = set()
        proies_temporaires = set()
        plantes_visibles = set()
        eau_visible = set()

        # Analyse des autres animaux
        for autre in autres_animaux:
            if autre is self:
                continue

            distance = self.distance(autre)
            dans_vision = self.est_dans_vision(autre)

            if dans_vision or distance < 50:
                if self.peut_chasser(autre):
                    proies_temporaires.add(autre)
                elif autre.peut_chasser(self):
                    predateurs_temporaires.add(autre)

        # Analyse des ressources
        for ressource in Ressources:
            distance = self.distance(ressource)
            dans_vision = self.est_dans_vision(ressource)

            if dans_vision or distance < 50:
                if ressource.type == "plante":
                    plantes_visibles.add(ressource)
                elif ressource.type == "eau":
                    eau_visible.add(ressource)

        # Mise à jour des attributs
        self.listpredateurs = list(predateurs_temporaires)
        self.listproies = list(proies_temporaires)
        self.listplantes = list(plantes_visibles)
        self.listeau = list(eau_visible)

        self.proies = len(self.listproies)
        self.predateurs = len(self.listpredateurs)
        self.plantes = len(self.listplantes)
        self.eaux = len(self.listeau)

    def peut_chasser(self, autre):
        """Détermine si cet animal peut chasser l'autre """
        relations_predation = {
            "lion": ["gazelle", "lapin"],
            "loup": ["lapin"],
            "ours": ["lapin", "gazelle"],
        }
        return autre.nom.lower() in relations_predation.get(self.nom.lower(), [])

    def eviter_les_bords(self, marge=50, urgence=False):
        """Modifie la direction pour éviter les bords de manière plus intelligente.
        
        Args:
            marge (int): Distance aux bords où le comportement s'active
            urgence (bool): Si True, évitement plus brutal (quand en danger immédiat)
        """
        largeur, hauteur = 10000, 10000  # Dimensions de l'environnement
        
        # Coefficients dynamiques selon l'urgence
        force_evitement = 1.2 if urgence else 0.7
        attenuation = 0.8 if urgence else 0.95  # Réduction des oscillations
        
        # Calcul des distances aux bords (normalisées entre 0 et 1)
        dist_x = min(self.x, largeur - self.x) / marge
        dist_y = min(self.y, hauteur - self.y) / marge
        
        # Évitement horizontal (plus intelligent)
        if self.x < marge:
            influence = (1 - dist_x) ** 2  # Influence quadratique
            self.dx += force_evitement * influence
        elif self.x > largeur - marge:
            influence = (1 - dist_x) ** 2
            self.dx -= force_evitement * influence
        
        # Évitement vertical
        if self.y < marge:
            influence = (1 - dist_y) ** 2
            self.dy += force_evitement * influence
        elif self.y > hauteur - marge:
            influence = (1 - dist_y) ** 2
            self.dy -= force_evitement * influence
        
        # Réduction des oscillations (effet de "frottement")
        self.dx *= attenuation
        self.dy *= attenuation
        
        # Limitation de vitesse plus progressive
        speed = math.sqrt(self.dx**2 + self.dy**2)
        if speed > self.vitesse_possible:
            ratio = (self.vitesse_possible / speed) ** 0.3  # Correction progressive
            self.dx *= ratio
            self.dy *= ratio

    def se_deplacer_aleatoire(self):
        """Déplacement aléatoire avec inertie directionnelle et recentrage vers le territoire."""

        # Variation aléatoire de l'angle (petite rotation)
        self.angle += random.uniform(-0.2, 0.2)

        # Force de rappel vers le centre
        vers_centre_x = self.territoire_x - self.x
        vers_centre_y = self.territoire_y - self.y
        distance_centre = math.hypot(vers_centre_x, vers_centre_y)
        angle_centre = math.atan2(vers_centre_y, vers_centre_x)

        # Si trop loin, orienter légèrement vers le centre
        if distance_centre > self.rayon_territoire:
            facteur_alignement = 0.05
            self.angle = (1 - facteur_alignement) * self.angle + facteur_alignement * angle_centre

        # Calcul du déplacement
        dx = math.cos(self.angle) * self.vitesse
        dy = math.sin(self.angle) * self.vitesse

        # Appliquer le déplacement avec vérification des bords
        self.x = max(0, min(self.x + dx, 20000))
        self.y = max(0, min(self.y + dy, 20000))

        self.eviter_les_bords()

        # Consommer de l'énergie
        self.consommer_energie("se_deplacer_aleatoire")

    def explorer_librement_ressources(self):
        """
        Exploration aléatoire sur la carte entière, sans considération du territoire,
        tout en gardant une trace des zones récemment visitées pour éviter de tourner en rond.
        """

        # Évite les zones récemment visitées
        for ancien_x, ancien_y in self.memoire_zones:
            distance = math.hypot(self.x - ancien_x, self.y - ancien_y)
            if distance < 200:  # Zone déjà explorée récemment
                self.angle += random.uniform(0.3, 0.5)  # Changement de direction

        # Petite variation aléatoire de direction
        self.angle += random.uniform(-0.1, 0.1)

        # Calcul du déplacement selon l’angle actuel
        dx = math.cos(self.angle) * self.vitesse
        dy = math.sin(self.angle) * self.vitesse

        # Appliquer le déplacement
        self.x += dx
        self.y += dy

        # Empêcher de sortir de la carte
        self.eviter_les_bords()

        # Mémoriser la nouvelle position
        self.memoire_zones.append((self.x, self.y))

        # Consommer de l'énergie
        self.consommer_energie("explorer_librement_nourriture")


    def se_deplacer_vers(self, cible):
        """
        Se rapproche d'une cible : ajuste l'angle, calcule dx et dy, applique le déplacement à la position.
        Réduction de la vitesse si elle dépasse un seuil.
        """

        # Réduction de la vitesse si elle dépasse un seuil de confort
        if self.vitesse > self.vitesse_possible * 0.6:
            self.vitesse *= 0.7
        else:
            self.vitesse *= self.acceleration
            

        # Calcule l'angle vers la cible
        self.angle = math.atan2(cible.y - self.y, cible.x - self.x)

        # Calcule dx et dy selon la vitesse et l'angle
        self.dx = math.cos(self.angle) * self.vitesse
        self.dy = math.sin(self.angle) * self.vitesse

        # Applique le déplacement
        self.x += self.dx
        self.y += self.dy

    def marcher_vers(self, cible):
        """
        Se rapproche d'une cible : ajuste l'angle, calcule dx et dy, applique le déplacement à la position.
        Réduction de la vitesse si elle dépasse un seuil.
        """

        # Réduction de la vitesse si elle dépasse un seuil de confort
        if self.vitesse > self.vitesse_possible * 0.3:
            self.vitesse *= 0.7

        # Calcule l'angle vers la cible
        self.angle = math.atan2(cible.y - self.y, cible.x - self.x)

        # Calcule dx et dy selon la vitesse et l'angle
        self.dx = math.cos(self.angle) * self.vitesse
        self.dy = math.sin(self.angle) * self.vitesse

        # Applique le déplacement
        self.x += self.dx
        self.y += self.dy
        
    def courir_vers(self, cible):
       
        # Augmenter progressivement la vitesse jusqu'à la vitesse max
        
        nouvelle_vitesse = min(self.vitesse + self.acceleration, self.vitesse_possible)
        self.vitesse = nouvelle_vitesse

        # Déplacement vers la cible
        self.se_deplacer_vers(cible)

        # Si trop fatigué, ralentir automatiquement
        if self.fatigue > 70:
            print(f"{self.nom} est fatigué et ralentit...")
            self.vitesse *= 0.8

    def recuperer_apres_course(self):
        """Ralentit progressivement et réduit la fatigue si l'animal n'est plus en action intense."""

        if self.vitesse > 3:  
            self.vitesse -= self.acceleration * 0.5
            if self.vitesse < 3:
                self.vitesse = 3

        if self.fatigue > 0:
            self.fatigue -= 0.8  # Repos progressif
    
    def fuir(self):
        """Fuit les prédateurs avec une fuite fluide et réaliste, en lissant les changements de direction."""
        if not self.listpredateurs:
            return  # Aucun prédateur à fuir

        # Trouver le prédateur le plus proche
        predateur_proche, distance = min(
            ((p, self.distance(p)) for p in self.listpredateurs),
            key=lambda item: item[1]
        )

        # Calcul de l'angle opposé au prédateur (fuite)
        angle_ennemi = self.angle_entre(predateur_proche)
        angle_fuite = angle_ennemi + math.pi  # Fuir dans la direction opposée
        self.angle=angle_fuite
        # Appliquer une variation aléatoire plus subtile
        variation_max = 0.3 * (1 - min(distance / 100, 0.6))  # max 30% de variation
        angle_fuite += random.uniform(-variation_max, variation_max)

        # Lisser l'angle pour éviter les changements brusques
        if hasattr(self, "angle_fuite_precedent"):
            delta = angle_fuite - self.angle_fuite_precedent

            # Normalisation de l'angle entre -pi et pi
            delta = (delta + math.pi) % (2 * math.pi) - math.pi

            # Limiter la variation par tick
            max_delta = 0.2  # radians (~11 degrés)
            delta = max(-max_delta, min(delta, max_delta))
            angle_fuite = self.angle_fuite_precedent + delta

        self.angle_fuite_precedent = angle_fuite  # Stocker pour le prochain tick

        # Distance de fuite dynamique
        distance_fuite = self.vitesse * self.acceleration
        if distance < 20:
            distance_fuite *= 1.5
        elif distance < 40:
            distance_fuite *= 1.2

        # Appliquer le déplacement
        self.x += math.cos(angle_fuite) * distance_fuite
        self.y += math.sin(angle_fuite) * distance_fuite


        # Éviter les bords avec plus d'urgence si le prédateur est proche
        self.eviter_les_bords(urgence=distance < 30)

    
    def chasser(self, proie):

        distance = self.distance(proie)
        if distance <= 5:
            self.mordre(proie)
            self.marcher_vers(proie)
        if distance < 70 and  distance > 5:
            self.courir_vers(proie)
        elif distance <= self.vision and  distance>50:
            self.marcher_vers(proie)

    def mordre(self, proie):

        if self.energie > 10:
            
            proie.subir_degats(20)  
            self.energie -= 2  

    def subir_degats(self, degats):
        """Subit des dégâts et réagit en conséquence."""
        self.energie -= degats  # Réduit la masse de l'animal (peut être lié à sa santé)
        if self.energie <= 0:
            print(f"{self.nom} est trop affaibli et meurt.")
            self.etat = "mort"  # L'animal est mort
            self.est_vivant = False
        
    def se_reposer(self):
        if( self.vitesse>2):
            self.vitesse *=0.8
        self.fatigue -= 4
        self.energie += 5
        if self.fatigue < 0:
            self.fatigue = 0
        if self.energie > 100:  
            self.energie = 100

    def mettre_a_jour_faim(self):
        """Met à jour le niveau de faim en fonction de la nourriture mangée."""
        # Si l'animal a mangé, la faim diminue
        self.faim -= self.nourriture_mangée * 10  
        self.soif-=self.nourriture_mangée*5
        if self.faim < 0:
            self.faim = 0

        if self.soif < 0:
            self.faim = 0

        if self.soif >100:
            self.soif = 100

        if self.nourriture_mangée == 0:
            self.faim += 0.001 

        self.soif += 0.01 
        
        if self.faim > 100:
            self.faim = 100

    def ajuster_vision(self, valeur):
        """Ajuste le champ de vision de l'animal."""
        self.vision = valeur
        print(f"{self.nom} a maintenant une vision de {self.vision} pixels.")

    def ajuster_angle_vision(self, valeur):
        """Ajuste l'angle de vision de l'animal."""
        self.angle_vision = valeur
        print(f"{self.nom} a maintenant un angle de vision de {self.angle_vision} degrés.")
  
    def consommer_energie(self, activite="marche", duree=1):
        """
        Consomme de l'énergie en fonction du poids, de la vitesse, de l'activité,
        de la fatigue accumulée, de la faim et de la soif.
        """

        # Facteurs d'activité
        facteurs_activite = {
            "se_reposer": 0.002,
            "marche": 0.05,
            "fuir": 0.22,
            "mordre": 0.08,
            "courir_vers": 0.01,
        }

        # Récupération du facteur d'activité (défaut = marche)
        F_activite = facteurs_activite.get(activite, 0.005)

        # Métabolisme de base
        metabolism_base = 0.5 * self.poids

        # Coût énergétique lié à la vitesse
        cout_vitesse = (self.vitesse ** 1.5) / 50

        # Fatigue accumulée (augmente le coût si effort prolongé)
        facteur_fatigue = 0.05 + 0.01 * self.fatigue

        # Calcul de la consommation énergétique
        consommation = (metabolism_base + cout_vitesse) * F_activite * facteur_fatigue * duree

        # Réduction de l’énergie disponible
        self.energie = max(0, self.energie - consommation)

        # Mise à jour de la fatigue
        if activite in ["courir_vers", "chasser", "fuir"]:
            self.fatigue += duree
        else:
            self.fatigue = max(0, self.fatigue - duree / 2)

        # Réévaluation de l’état
        if self.fatigue >= 70:
            self.etat = "fatigué"

        # --- Ajustement énergétique dû à la faim et la soif ---
        # Plus faim/soif est élevée (au-dessus de 50%), plus l'efficacité énergétique baisse
        faim_facteur = max(0, (self.faim - 50) / 50) ** 1.2
        soif_facteur = max(0, (self.soif - 50) / 50) ** 1.1

        penalite_faim = 40 * faim_facteur
        penalite_soif = 30 * soif_facteur

        self.energie = max(0, self.energie - penalite_faim - penalite_soif)

    def regrouper(self):
        self.fuir()
        return
    
    def explorer(self):
        self.se_deplacer_aleatoire()
        return
    
    def mise_a_jour(self, autres_animaux, ressouces):
        """
        Méthode appelée à chaque tick de la simulation.
        Elle met à jour les états vitaux du lion sans prendre de décision comportementale.
        """
        self.mettre_a_jour_environnement(autres_animaux, ressouces)
        self.mettre_a_jour_faim()
        self.vitesse_max_atteignable()
        self.recuperer_apres_course()
        if (len(self.memoire_actions)==3):
            self.consommer_energie(self.memoire_actions[0])


    def attribuer_couleur(self):
        couleurs = {
            "ours": "#8B4513",    # Marron
            "lion": "#FFD700",    # Or
            "gazelle": "#DAA520", # Doré
            "lapin": "#FFFFFF",   # Blanc
        }
        return couleurs.get(self.nom.lower(), "#808080")  # Gris par défaut si inconnu
    
    def attribuer_vitesse(self):
        vitesses = {
            "ours": 2,
            "lion": 4,
            "gazelle": 6,
            "lapin": 5,
        }
        return vitesses.get(self.nom.lower(), 3)  # Par défaut 3

    def attribuer_vision(self):
        visions = {
            "ours": 80,
            "lion": 120,
            "gazelle": 150,
            "lapin": 130,
        }
        return visions.get(self.nom.lower(), 100)  # Par défaut 100

    def attribuer_angle_vision(self):
        angles = {
            "ours": 100,
            "lion": 120,
            "gazelle": 160,
            "lapin": 170,
        }
        return angles.get(self.nom.lower(), 90)

    def attribuer_acceleration(self):
        accels = {
            "ours": 1.5,
            "lion": 2.5,
            "gazelle": 3,
            "lapin": 2.8,
        }
        return accels.get(self.nom.lower(), 2)

    def attribuer_vitesse_max(self):
        vmax = {
            "ours": 20,
            "lion": 40,
            "gazelle": 50,
            "lapin": 45,
        }
        return vmax.get(self.nom.lower(), 30)



# Méthodes vital
    

    def dormir(self):
        self.vitesse*=0.01
        return
    
    def boir(self):
        return
    
    def se_cacher(self):
        self.fuir()
        return
    
    def rester_alerte(self):
        """Maintien en état de vigilance"""
        pass
    
    # Méthodes environnementales
    def chercher_abri(self):
        """Recherche un abri contre les intempéries"""
        pass
    
    # Méthodes physiologiques
    def boire(self):
        """Action de boire de l'eau"""
        pass
        
    def chercher_eau(self):
        """Recherche d'une source d'eau"""
        pass
        
    def manger(self):
        """Consommation de nourriture disponible"""
        pass
        
    def chasser(self):
        """Chasse des proies pour se nourrir"""
        pass
        
    def chercher_nourriture(self):
        """Recherche de sources de nourriture"""
        pass
    
    # Méthodes de repos
    def se_reposer(self):
        """Récupération d'énergie"""
        pass
    
    # Méthodes sociales
    def se_reproduire(self):
        """Accouplement avec un partenaire"""
        pass
        
    def chercher_partenaire(self):
        """Recherche d'un partenaire sexuel"""
        pass
    
    # Méthode par défaut
    def explorer(self):
        """Exploration de l'environnement"""
        pass







class Ressource:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Plante(Ressource):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "plante"

    def __repr__(self):
        return f"Plante({self.x:.0f}, {self.y:.0f})"


class Eau:
    def __init__(self, x, y, rayon):
        self.x = x
        self.y = y
        self.rayon = rayon
        self.type = "eau"
        self.points = self.generer_points()

    def generer_points(self):
        points = []
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, self.rayon)
            px = self.x + math.cos(angle) * r
            py = self.y + math.sin(angle) * r
            if 0 <= px <= 10000 and 0 <= py <= 10000:
                points.append((px, py))
        return points