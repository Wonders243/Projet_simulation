
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
with open(encoder_decision_path, 'rb') as f:
    encoder_decision = pickle.load(f)

class Simulation (AsyncWebsocketConsumer):
    largeur = 1800  
    hauteur = 620 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_layer = get_channel_layer()
        self.largeur = 728  # Largeur du canvas en pixels
        self.hauteur = 389  # Hauteur du canvas en pixels
        self.territoires = {
            'lion': (320, 150),
            'lapin': (80, 300),
            'gazelles': (600, 200),
            'ours': (450, 50)
        }
        
        self.annee = 2025
        self.mois = 3
        self.jour = 25
        self.heure = 12
        self.minute = 0
        self.seconde = 0

        self.temperature=20
        self.climat= "neige"
        self.saison=""

        self.tick_duree = 0.01  # 10 ms par tick
        self.ticks_par_heure = 200  # 1 heure simulée = 20 secondes réelles
        self.ticks_par_minute = 10  # 1 minute simulée = 33 ticks
        self.ticks_par_jour = 48000  # 1 jour simulé = 48000 ticks = 8 minutes réelles
        self.ticks_actuels =0
        # Liste des animaux présents dans la simulation
        self.animaux = []
        self.Old_animaux = []

        self.ressources = self.generer_ressources()

        # Création et ajout de quelques animaux dans la simulation
        self.initialiser_animaux()
    
    def initialiser_animaux(self):
        """ Initialise quelques animaux dans la simulation. """
        # Ajouter des animaux à la simulation avec leurs caractéristiques
        animaux_types = ["lapin", "lion", "ours"]
        for _ in range(20):
            type_animal = random.choice(animaux_types)
            age = random.randint(1, 15)
            poids = random.randint(10, 200)
            energie = random.randint(60, 100)
            faim = random.randint(0, 30)
            soif = random.randint(0, 31)
            y,x= self.territoires[type_animal]
            if type_animal == "lion":
                territoire = self.territoires['lion']
            elif type_animal == "lapin":
                territoire = self.territoires['lapin']
            elif type_animal == "ours":
                territoire = self.territoires['ours']
            else:
                territoire = {"x": 1000, "y": 1000}  # Par défaut, si un type inconnu (par sécurité)



            self.animaux.append(Animal(type_animal, x=x, y=y, age=age, poids=poids, energie=energie, faim=faim, soif=soif, territoire=territoire))

    def generer_ressources(self, test=100, n_zones_eau=10):
        ressources = []
        for _ in range(n_zones_eau):
            x = random.randint(0, Simulation.largeur)
            y = random.randint(0, Simulation.hauteur)
            rayon = random.randint(150, 300)
            ressources.append(Eau(x, y, rayon))

        for _ in range(test):
            x = random.randint(0, Simulation.largeur)
            y = random.randint(0, Simulation.hauteur)
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

        # Calcul du temps écoulé en secondes simulées
        secondes_simulees = self.ticks_actuels / self.ticks_par_minute * 60
        
        # Conversion en composantes de temps
        self.seconde = int(secondes_simulees % 60)
        self.minute = int((secondes_simulees // 60) % 60)
        self.heure = int((secondes_simulees // 3600) % 24)
        
        # Calcul des jours/mois/années
        jours_simules = self.ticks_actuels / self.ticks_par_jour
        self.jour = int(jours_simules % 30) + 1
        self.mois = int(jours_simules // 30) % 12 + 1
        self.annee = 2025 + int(jours_simules // 365)

        # Détermination de la saison
        self.saison = self.determiner_saison()
        
        # Mise à jour du climat en fonction de la saison
        self.mettre_a_jour_climat()
        
        self.cycle = "Jour" if self.est_jour() else "Nuit"

    def determiner_saison(self):
        if 3 <= self.mois <= 5:
            return "Printemps"
        elif 6 <= self.mois <= 8:
            return "Été"
        elif 9 <= self.mois <= 11:
            return "Automne"
        else:  # mois 12, 1, 2
            return "Hiver"

    def mettre_a_jour_climat(self):
        # Température de base selon la saison
        temp_saison = {
            "Printemps": 15,
            "Été": 30,
            "Automne": 10,
            "Hiver": -5
        }
        
        # Variation aléatoire de température
        variation = random.uniform(-2, 2)
        self.temperature = temp_saison[self.saison] + variation
        
        # Détermination du climat en fonction de la température
        if self.temperature > 25:
            self.climat = "ensoleillé"
        elif self.temperature > 15:
            self.climat = "nuageux"
        elif self.temperature > 5:
            self.climat = "pluie"
        elif self.temperature > 0:
            self.climat = "vent"
        else:
            self.climat = "neige"
            
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
            predateurs = animal.predateurs  # Nombre de prédateurs
            partenaire_proche=animal.partenaire_proche
            pret_reproduction=animal.pret_reproduction

            # Ajout des données de l'animal dans la liste sous forme de ligne
            donnees_animaux.append([
                nom, age, poids, energie, faim, soif, nourriture_dispo, eau_proche, 
                 predateurs, heure, pret_reproduction, partenaire_proche
            ])

        # Convertir la liste de données en DataFrame pandas
        df = pd.DataFrame(donnees_animaux, columns = ["animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", "predateurs", "heure",  "pret_reproduction", "partenaire_proche"])
       
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
            
            # Besoins physiologiques
            
            "chasser": animal.chasser,
            
            # Reproduction
            "se reproduire": animal.se_reproduire,
            "chercher un partenaire": animal.explorer,
            
            # Comportements
            "se reposer": animal.se_reposer,
            "dormir": animal.dormir,
            "explorer": animal.explorer,
            "jouer / interagir": animal.jouer_interagir,
            
            # Actions composites
            "satisfaire besoin": self._satisfaire_besoin,
            "chercher ressources": animal.se_deplacer_aleatoire
        }
        
        # Exécution de l'action
        action_finale = animal.determiner_action_finale(action)
        
        if action_finale in actions_disponibles:
            if action_finale in ["satisfaire besoin"]:
                actions_disponibles[action_finale](animal)
            elif action_finale in ["se reproduire"]:
                actions_disponibles[action_finale](self.animaux)
            else:
                actions_disponibles[action_finale]()
        else:
            animal.explorer()  # Action par défaut


    def _satisfaire_besoin(self, animal):
        """Gère la satisfaction des besoins (eau ou nourriture)"""
        if animal.soif > animal.faim:
            if animal.eau_proche==1:
                animal.boir()
            else:
                self.ressources= animal.explorer_ressources(self.ressources, self.animaux)
        else:
            if animal.nourriture_dispo==1:
                animal.manger()
            elif animal in ["lion, ours"]:
               self.ressources= animal.explorer_ressources(self.ressources, self.animaux)


    async def demarrer(self):

        self.Old_animaux=self.animaux
        vie= len(self.Old_animaux)>0
        while vie:
            self.avancer_temps()
            donnees=self.recuperer_donnees_animaux()
            actions= self.animal_Action(donnees)
            print(donnees)
            print(actions)

            for i, animal in enumerate(self.animaux) :
                if(i==len(actions)):
                    break
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
        self.dx =0.3
        self.dy=0.3
        self.age = age
        self.poids = poids
        self.energie =energie
        self.vitesse_possible=self.vitesse_max
        self.faim = faim  
        self.soif = soif 
        self.eau_bu = 0 
        self.nourriture_dispo = 0  
        self.eau_proche = 0 
        self.predateurs = 0
        self.proies = 0
        self.listpredateurs = []
        
        self.listproies = []
        self.plantes = [] 
        self.eaux = []
        self.nourriture = []
        self.vision = 100  
       
        self.etat = "actif" 
        self.partenaire_proche=0
        self.pret_reproduction=random.choice([0, 1])
        self.angle = 0  

        self.fatigue=0
        self.nourriture_mangée=0
        self.vitesse=vitesse
        self.territoire_x, self.territoire_y = territoire
        self.rayon_territoire = rayon_territoire
        self.memoire_actions =  deque(maxlen=3) 
        self.memoire_zones = []
        self.est_vivant = True

    def mourir(self):
        self.etat="inactif"
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
        reduction_energie = (100 - self.energie) / 500

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
        self.eau_proche= len(self.listeau)

    def peut_chasser(self, autre):
        """Détermine si cet animal peut chasser l'autre """
        relations_predation = {
            "lion": ["gazelle", "lapin"],
            "loup": ["lapin"],
            "ours": ["lapin", "gazelle"],
        }
        return autre.nom.lower() in relations_predation.get(self.nom.lower(), [])
   
    def eviter_les_bords(self, marge=50, urgence=True):
        """Modifie uniquement la direction pour éviter les bords (pas d'atténuation ni de ralentissement)."""
        largeur = Simulation.largeur
        hauteur = Simulation.hauteur

        force_evitement = 1.2 if urgence else 0.7

        # Évitement horizontal
        if self.x < marge:
            self.dx += force_evitement
        elif self.x > largeur - marge:
            self.dx -= force_evitement

        # Évitement vertical
        if self.y < marge:
            self.dy += force_evitement
        elif self.y > hauteur - marge:
            self.dy -= force_evitement

        # Limitation douce de la vitesse (juste pour éviter les excès)
        speed = math.sqrt(self.dx**2 + self.dy**2)
        if speed > self.vitesse_possible:
            ratio = self.vitesse_possible / speed
            self.dx *= ratio
            self.dy *= ratio

        # Forcer la position dans les limites si nécessaire
        self.x = max(0, min(self.x, largeur))
        self.y = max(0, min(self.y, hauteur))

    def se_deplacer_aleatoire(self):
        self.etat = "actif"
        """Déplacement aléatoire libre, sans attraction vers un territoire."""

        # Changement d'angle plus ou moins brusque selon un facteur aléatoire
        if random.random() < 0.005:  
            self.angle = random.uniform(0, 2 * math.pi)  # Nouvelle direction aléatoire
        else:
            self.angle += random.uniform(-0.2, 0.2)  # Dérive progressive

        # Variation de vitesse pour plus de naturel
        vitesse_actuelle = self.vitesse * random.uniform(0.8, 1.2)

        # Calcul du déplacement
        dx = math.cos(self.angle) * vitesse_actuelle
        dy = math.sin(self.angle) * vitesse_actuelle

        self.x += dx
        self.y += dy

        self.eviter_les_bords(urgence=True)
        
        
    def explorer_librement_ressources(self, ressources):
        self.etat="actif"
        """
        Exploration aléatoire de la carte, tout en vérifiant la présence de ressources
        dans le champ de vision. Met à jour l'état si des ressources sont détectées.
        
        :param ressources: Liste de tuples (type_ressource, (x, y))
        """
        if not isinstance(ressources, list):
            return
            
        # Limiter la mémoire des zones visitées
        if len(self.memoire_zones) > 50:
            self.memoire_zones.pop(0)
        
        # Vérifier les ressources visibles
        for ressource in ressources:
            if not hasattr(ressource, 'x') or not hasattr(ressource, 'y'):
                continue
                
            type_ressource = "eau" if isinstance(ressource, Eau) else "nourriture"
            position=Position(ressource.x, ressource.y)

            if self.est_dans_vision(position):
                if type_ressource == 'eau' and position not in self.eaux:
                    self.eau_proche = 1
                    self.eaux.append(position)
                elif type_ressource == 'nourriture' and position not in self.nourriture:
                    self.nourriture_dispo = 1
                    self.nourriture.append(position)

        # Évite les zones récemment visitées avec une probabilité croissante
        for ancien_x, ancien_y in self.memoire_zones[-10:]:  # Ne vérifier que les 10 dernières
            distance = math.hypot(self.x - ancien_x, self.y - ancien_y)
            if distance < 200:
                # Plus on est proche, plus on a de chance de changer de direction
                if random.random() < (200 - distance) / 200:
                    self.angle += random.uniform(0.3, 0.5)

        # Variation aléatoire de direction mais avec une tendance à aller tout droit
        self.angle += random.uniform(-0.1, 0.1)
        
        # Normaliser l'angle pour éviter les valeurs trop grandes
        self.angle %= 2 * math.pi

        # Calcul du déplacement
        dx = math.cos(self.angle) * self.vitesse
        dy = math.sin(self.angle) * self.vitesse

        # Appliquer le déplacement
        self.x += dx
        self.y += dy

        # Empêcher de sortir de la carte
        self.eviter_les_bords()

        # Mémoriser la position
        self.memoire_zones.append((self.x, self.y))
   
    def se_deplacer_vers(self, cible):
        self.etat="actif"
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
        self.angle = self.angle_entre(cible)

        # Calcule dx et dy selon la vitesse et l'angle
        self.dx = math.cos(self.angle) * self.vitesse
        self.dy = math.sin(self.angle) * self.vitesse

        # Applique le déplacement
        self.x += self.dx
        self.y += self.dy

    def marcher_vers(self, cible):
        self.etat="actif"
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
        self.eviter_les_bords()
        
    def courir_vers(self, cible):
        self.etat = "actif"
        
        # 1. Calcul de la direction vers la cible
        angle_vers_cible = self.angle_entre(cible)

        # 2. Calcul des composantes dx/dy avec acceleration
        self.dx += math.cos(angle_vers_cible) * self.acceleration
        self.dy += math.sin(angle_vers_cible) * self.acceleration
      
        # 4. Limitation de vitesse
        vitesse_totale = math.sqrt(self.dx**2 + self.dy**2)
        if vitesse_totale > self.vitesse_possible:
            ratio = self.vitesse_possible / vitesse_totale
            self.dx *= ratio
            self.dy *= ratio
        
        # 5. Application du déplacement
        self.x += self.dx
        self.y += self.dy
        
        # 6. Gestion de la fatigue
        if self.fatigue > 70:
            self.dx *= 0.9  # Ralentissement progressif
            self.dy *= 0.9
            self.vitesse_possible = max(2, self.vitesse_possible * 0.9)  # Vitesse max réduite
        
        # 7. Mise à jour de l'angle (pour les autres méthodes)
        self.angle = angle_vers_cible
        self.eviter_les_bords()

    def recuperer_apres_course(self):
        """Ralentit progressivement et réduit la fatigue si l'animal n'est plus en action intense."""

        if self.vitesse > 3:  
            self.vitesse -= self.acceleration * 0.5
            if self.vitesse < 3:
                self.vitesse = 3

        if self.fatigue > 0:
            self.fatigue -= 0.8  # Repos progressif
        
    def fuir(self):


        if not self.listpredateurs:
            return  

        # Trouver le prédateur le plus proche
        predateur_proche = min(self.listpredateurs, key=lambda p: self.distance(p))

        # Calcul des différences
        dx = self.x - predateur_proche.x
        dy = self.y - predateur_proche.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist > 0:
            # Accélération dans la direction opposée au prédateur
            self.dx += 0.1*(dx / dist) * self.acceleration
            self.dy += 0.1*(dy / dist) * self.acceleration

            # Limitation de la vitesse maximale
            speed = math.sqrt(self.dx ** 2 + self.dy ** 2)
            if speed > self.vitesse_max:
                self.dx = self.vitesse_max
                self.dy = self.vitesse_max

        # Mise à jour de la position
        self.x += self.dx
        self.y += self.dy
            

    def chercher_abri():
        pass

    def chasser(self):

        if len(self.listproies) > 0:
            proie_proche = min(self.listproies, key=lambda a: self.distance(a))
          
            distance = self.distance(proie_proche)
            if distance == 0:
                self.mordre(proie_proche)
                self.marcher_vers(proie_proche)
            else:
                self.courir_vers(proie_proche)
        else:
            self.explorer()
            

    def mordre(self, proie):
        self.etat="actif"
        if self.energie > 10:
            
            proie.subir_degats(20)  
            self.energie -= 2  
           
        if(self.energie<0) :
             self.energie=0 

    def subir_degats(self, degats):
        """Subit des dégâts et réagit en conséquence."""
        self.energie -= degats  # Réduit la masse de l'animal (peut être lié à sa santé)
        if self.energie <= 0:
            print(f"{self.nom} est trop affaibli et meurt.")
            self.etat = "mort"  # L'animal est mort
            self.est_vivant = False
        if(self.energie<0) :
             self.energie=0 
    
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
  
    def consommer_energie(self):
        
        # Métabolisme de base
        metabolism_base = 0.05 * self.poids

        # Coût énergétique lié à la vitesse
        cout_vitesse = (self.vitesse ** 1.8) / 10

        # Fatigue accumulée (augmente le coût si effort prolongé)
        facteur_fatigue = 0.01 + 0.05 * self.fatigue

        # Calcul de la consommation énergétique
        consommation = 0.09*(metabolism_base + cout_vitesse)* facteur_fatigue 

        # Réduction de l’énergie disponible
        self.energie = max(0, self.energie - consommation)
  

    def regrouper(self):
        self.fuir()
        return
    
    def explorer_ressources(self, ressources, autres):
        """Explore l'environnement et classe les ressources visibles par type"""
        # Réinitialiser les listes de ressources visibles
        self.plantes = []
        self.eaux = []
        self.nourriture = []
        
        # Déplacement aléatoire pour l'exploration
        self.se_deplacer_aleatoire()
        
        
        # Parcourir toutes les ressources disponibles
        for ressource in ressources:
            # Calculer la distance
            
            distance = self.distance(Position(ressource.x, ressource.y))
            
            # Vérifier si la ressource est proche et visible
            if distance <= 300 and self.est_dans_vision(Position(ressource.x, ressource.y)):
                
                if ressource.type == "plante":
                    self.plantes.append(ressource)
                elif ressource.type == "eau":
                    self.eaux.append(ressource)
                elif ressource.type == "nourriture":
                    self.nourriture.append(ressource)
        
        # Mettre à jour les indicateurs de disponibilité
        self.nourriture_dispo = 1 if (len(self.nourriture) > 0) else 0
        self.eau_proche = 1 if(len(self.eaux) > 0) else 0
        
        # Retourner les ressources (non modifiées dans cet exemple)
        return ressources
    
    def mise_a_jour(self, autres_animaux, ressources):
        """
        Méthode appelée à chaque tick de la simulation.
        Elle met à jour les états vitaux du lion sans prendre de décision comportementale.
        """
        if(random.random()<0.01) and self.energie>70:
            self.pret_reproduction=1
        if(random.random()<0.01):
            self.partenaire_proche=1

        self.mettre_a_jour_environnement(autres_animaux, ressources)
        self.mettre_a_jour_faim()
        self.vitesse_max_atteignable()
        self.recuperer_apres_course()
        self.consommer_energie()

    def chercher_partenaire(self):
        self.etat="actif"
        """Comportement de recherche de partenaire"""
        print(f"{self.nom} cherche un partenaire")
        # Comportement de recherche spécifique
        self.se_deplacer_aleatoire()

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
            "ours": 1.1,
            "lion": 1.2,
            "gazelle": 1.4,
            "lapin": 1.3,
        }
        return accels.get(self.nom.lower(), 2)

    def attribuer_vitesse_max(self):
        vmax = {
            "ours": 5,
            "lion": 7.5,
            "gazelle": 12,
            "lapin": 10,
        }
        return vmax.get(self.nom.lower(), 10)

    def jouer_interagir(self):
        pass

# Méthodes vital
    

    def dormir(self):
        position=Position(self.territoire_x, self.territoire_y)
        self.se_deplacer_vers(position)
        if(self.distance(position)<random.choice[0,300]):
            self.etat="invisible"
    
    def boir(self):
        """Comportement complet de boisson avec gestion de la soif, de l'énergie et recherche d'eau."""
       
        self.etat = "actif"
        if not self.eaux:
            return  
        point_eau = min(self.eaux, key=lambda eau: self.distance(eau.position))
        dist_eau = self.distance(point_eau + point_eau.rayon-5)
        
        if dist_eau < 5:  # Arrivé à l'eau
            
            self.eau_bu += min(5,self.soif/20)
            
                
        elif dist_eau < 500:  # Marche vers l'eau
            self.marcher_vers(point_eau)
            if point_eau not in self.memoire_zones:
                self.memoire_zones.append(point_eau)
                
        else: 
            if self.energie > 44: 
                self.courir_vers(point_eau)
                self.angle = self.angle_entre(point_eau)
            else:
                self.marcher_vers(point_eau)  
        if self.eau_bu> 0.6 *self.soif :
            self.energie+= self.eau_bu*0.3
            if self.energie>100:
                self.energie=100
                
    def manger(self):
       
        pass
    
    def se_cacher(self):
        position=Position(self.territoire_x, self.territoire_y)
        if (self.distance(position)<self.distance(position)>20):
            self.courir_vers(position)
        else:
            self.se_deplacer_vers(position)

        if(self.distance(position)<self.distance(position)<random.choice[0,20]):
            self.etat="invisible"
            self.vitesse=0
        
    
    def rester_alerte(self):
        """Maintien en état de vigilance"""
        pass
   
        
   
    # Méthodes de repos
    def se_reposer(self):
        position=Position(self.territoire_x, self.territoire_y)
        distance= self.distance(position)
        if(distance>500):
            self.se_deplacer_vers(position)
        else:
            self.marcher_vers(position)

        if(self.distance(position)<random.choice([0,300]) and self.etat!="repos" and self.vitesse>0):
            self.etat="repos"
            if( self.vitesse>2):
                self.vitesse *=0.8
            self.fatigue -= 4
            self.energie += 10
            if self.fatigue < 0:
                self.fatigue = 0
            if self.energie > 100:  
                self.energie = 100

    
    # Méthodes sociales
    def se_reproduire(self, animaux):
        """Crée un bébé animal et l'ajoute à la liste des animaux"""
            
        bebe = type(self)(  # Création dynamique avec le même type que le parent
            nom=self.nom,
            x=self.x + random.uniform(-10, 10),  # Petit décalage aléatoire
            y=self.y + random.uniform(-10, 10),
            age=0,
            poids=self.poids * 0.7,
            energie=self.energie * 0.8,
            faim=0.1,
            soif=0.1,
            territoire=(self.territoire_x, self.territoire_y),
            rayon_territoire=self.rayon_territoire
        )
        if(random.random()<0.01):
            animaux.append(bebe)
            self.energie *= 0.7  
            self.pret_reproduction = 0  

        return animaux
        
    def chercher_partenaire(self):
        self.marcher_vers(Position(self.territoire_x, self.territoire_y))

    # Méthode par défaut
    def explorer(self):
        """Exploration de l'environnement"""
        self.se_deplacer_aleatoire()
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
        self.x=x
        self.y=y
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

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    