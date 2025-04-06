
import random
import json
import math
import random
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
with open(encoder_climat_path, 'rb') as f:
    encoder_climat = pickle.load(f)
with open(encoder_decision_path, 'rb') as f:
    encoder_decision = pickle.load(f)

class Simulation (AsyncWebsocketConsumer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_layer = get_channel_layer()
        self.largeur = 20000  # Largeur du canvas en pixels
        self.hauteur = 20000  # Hauteur du canvas en pixels
        
        self.annee = 2025
        self.mois = 3
        self.jour = 25
        self.heure = 12
        self.minute = 0
        self.seconde = 0

        self.temperature=20
        self.climat= "neige"

        self.tick_duree = 0  # 50 ms par ticks
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
        for _ in range(60):
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

    def recuperer_donnees_animaux_transferer(self):
        """
        Cette méthode récupère toutes les données des animaux sous forme de tableau (DataFrame),
        en combinant les attributs des animaux avec ceux de la simulation.
        """
        # Liste pour stocker les données des animaux sous forme de liste
        donnees_animaux = []

        # Récuperation des donnees globales de la simulation
        temperature = self.temperature 
        climat = self.climat 
        heure = self.heure 
       
        # On parcourt chaque animal et on recupere ses donnees specifiques
        for animal in self.animaux:
            
            nom = animal.nom
            age = animal.age
            poids = animal.poids
            energie = animal.energie
            faim = animal.faim
            soif = animal.soif
            nourriture_dispo = animal.nourriture_dispo  
            eau_proche = animal.eau_proche  
            proies = animal.proies  
            predateurs = animal.predateurs  
            x=animal.x
            y=animal.y
            color= animal.color

            # Ajout des données de l'animal dans la liste sous forme de ligne
            donnees_animaux.append([
                nom, age, poids, energie, faim, soif, nourriture_dispo, eau_proche, 
                temperature, climat, predateurs, proies, heure, x, y, color
            ])

        # Convertir la liste de données en DataFrame pandas
        df = pd.DataFrame(donnees_animaux, columns=[
            "animal", "age", "poids", "energie", "faim", "soif", "nourriture", "eau", 
            "temperature", "climat", "predateurs", "proies", "heure", "x","y", "color"
        ])

        return df
    
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
        donnees = self.recuperer_donnees_animaux_transferer()  # Récupère les données en DataFrame
        
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
        await self.send(text_data=json.dumps(message))  # Envoi des données JSON au frontend

    def interpreter(self,animal, action):
        """ Interprète l'action et appelle la méthode correspondante sur l'animal. """
        
        if action == "chasser":
            if len(animal.listproies)>0:
                proie_proche = min(animal.listproies, key=lambda animal: animal.distance(animal))
                animal.chasser(proie_proche)
        elif action == "chercher de la nourriture":
            animal.chercher_nourriture()
        elif action == "boire":
            animal.boire()
        #elif action == "chercher de l'eau":
           # animal.chercher_eau()
        elif action == "se cacher":
            animal.se_cacher()
        elif action == "fuir":
            animal.fuir()
        elif action == "se regrouper":
            animal.regrouper()
        elif action == "dormir":
            animal.dormir()
        elif action == "se reposer":
            animal.se_reposer()
         #elif action == "jouer / interagir":
             #animal.jouer_interagir()
        else:  # Par défaut, si aucune condition spécifique n'est remplie
            animal.explorer()


    async def demarrer(self):
        while True:
            self.avancer_temps()
            for animal in self.animaux :
                animal.mettre_a_jour_environnement( self.animaux)
                animal.mettre_a_jour_faim()

            donnees=self.recuperer_donnees_animaux()
            actions= self.animal_Action(donnees)
            print(actions)
            for i, animal in enumerate(self.animaux):
                if(animal.est_vivant==True):
                    action = str(actions[i]).lower()
                    animal.recuperer_apres_course()
                    self.interpreter(animal, action) 
            await self.envoyer_donnees()
            await asyncio.sleep(self.tick_duree) 
     
            for animal in self.animaux:
                animal.se_deplacer_aleatoire()
                
            await asyncio.sleep(self.tick_duree)  # Pause de 50ms entre chaque tick
     
    


class Animal:
    def __init__(self, nom, x, y, age, poids, energie, faim, soif, vitesse=1, vision=100, angle_vision=90):
        self.nom = nom
        self.x = x 
        self.y = y
        self.dx =1
        self.dy=1
        self.age = age
        self.poids = poids
        self.energie = energie  # (0-100)
        self.vitesse_max=50
        self.faim = faim  # (0-100)
        self.soif = soif  # (0-100)
        self.nourriture_dispo = 0  # Nourriture proche
        self.eau_proche = 0  # 1 si eau proche, 0 sinon
        self.predateurs = 0
        self.proies = 0
        self.listpredateurs = []
        self.listproies = []
        self.vitesse = 1  # Vitesse de déplacement
        self.vision = 50  
        self.angle_vision = 45  
        self.etat = "actif" 
        self.angle = 0  # Direction de l'animal (en radians)
        self.acceleration=0.3
        self.color= self.attribuer_couleur()
        self.fatigue=0
        self.nourriture_mangée=0
        self.est_vivant = True

    def mourir(self):
        self.etat = "mort"
        self.est_vivant = False
        print(f"{self.nom} est mort.")

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
        return angle_diff <= self.angle_vision / 2 and self.distance(autre) <= self.vision

    def mettre_a_jour_environnement(self, autres_animaux):
        """Met à jour la liste des prédateurs et des proies visibles."""
        #self.listproies = []
        #self.listpredateurs = []
        for autre in autres_animaux:
            if autre is not self and self.est_dans_vision(autre) and autre :
                if self.peut_chasser(autre) and not self.listproies.__contains__(autre):  # Définir les relations proie/prédateur
                    self.listproies.append(autre)
                elif autre.peut_chasser(self) and not self.listpredateurs.__contains__(autre):
                    self.listpredateurs.append(autre)

            self.proies=len(self.listproies)
            self.predateurs = len(self.listpredateurs)

    def peut_chasser(self, autre):
        """Détermine si cet animal peut chasser l'autre """
        relations_predation = {
            "lion": ["gazelle", "lapin"],
            "loup": ["lapin"],
            "ours": ["lapin", "gazelle"],
        }
        return autre.nom.lower() in relations_predation.get(self.nom.lower(), [])

    def se_deplacer_vers(self, cible):
        """Se rapproche d'une cible avec déplacement progressif."""
        angle = math.atan2(cible.y - self.y, cible.x - self.x)
        self.x += math.cos(angle) * self.vitesse
        self.y += math.sin(angle) * self.vitesse
    
    def eviter_les_bords(self, marge=50):
        """Modifie la direction pour éviter de rester coincé aux bords de la carte."""
        largeur, hauteur = 2010, 2010  # Dimensions de la grille +1 pour cohérence avec range max

        if self.x < marge:
            self.dx += 0.5  # Tendance à aller vers la droite
        elif self.x > largeur - marge:
            self.dx -= 0.5  # Tendance à aller vers la gauche

        if self.y < marge:
            self.dy += 0.5  # Tendance à descendre
        elif self.y > hauteur - marge:
            self.dy -= 0.5  # Tendance à monter

        # Applique une limitation de vitesse douce
        speed = math.sqrt(self.dx**2 + self.dy**2)
        if speed > self.vitesse_max:
            self.dx *= self.vitesse_max / speed
            self.dy *= self.vitesse_max / speed


    def se_deplacer_aleatoire(self):
        """Déplacement aléatoire fluide en pixels avec inertie et évitement des bords."""
        self.dx += random.uniform(-0.2, 0.2)
        self.dy += random.uniform(-0.2, 0.2)
        self.eviter_les_bords()  
        self.x = max(0, min(self.x + self.dx, 20000))
        self.y = max(0, min(self.y + self.dy, 20000))
        self.consommer_energie("se_deplacer_aleatoire")

    def marcher_vers(self, cible):
        self.se_deplacer_vers(cible)
        
    def courir_vers(self, cible):
       
        # Augmenter progressivement la vitesse jusqu'à la vitesse max
        nouvelle_vitesse = min(self.vitesse + self.acceleration, self.vitesse_max)
        self.vitesse = nouvelle_vitesse

        # Déplacement vers la cible
        self.se_deplacer_vers(cible)

        # Fatigue : plus on va vite, plus ça fatigue
        fatigue_générée = self.vitesse * 0.1
        self.fatigue += fatigue_générée
        self.energie -= fatigue_générée * 0.5  # Bonus : courir consomme aussi de l'énergie

        # Limite fatigue (facultatif)
        if self.fatigue > 100:
            self.fatigue = 100

        # Si trop fatigué, ralentir automatiquement
        if self.fatigue > 70:
            print(f"{self.nom} est fatigué et ralentit...")
            self.vitesse *= 0.7  # Ralentissement si trop fatigué

        self.consommer_energie("courir_vers")

    def recuperer_apres_course(self):
        """Ralentit progressivement et réduit la fatigue si l'animal n'est plus en action intense."""

        if self.vitesse > 3:  
            self.vitesse -= self.acceleration * 0.5
            if self.vitesse < 3:
                self.vitesse = 3

        if self.fatigue > 0:
            self.fatigue -= 0.5  # Repos progressif



    def fuir(self):
        """Fuit les prédateurs en s'éloignant du plus proche."""
        
        # Trouver le prédateur le plus proche
        predateur_proche = min(self.listpredateurs, key=lambda p: self.distance(p))

        angle = math.atan2(self.y - predateur_proche.y, self.x - predateur_proche.x)
        
        self.x += math.cos(angle) * self.vitesse * 2  
        self.y += math.sin(angle) * self.vitesse * 2
        
        # Consommer de l'énergie en fuyant
        self.consommer_energie("fuir")

    def chasser(self, proie):

        distance = self.distance(proie)
        if distance <= 5:
            self.mordre(proie)
            self.marcher_vers(proie)
        if distance < 70 and  distance > 5:
            self.courir_vers(proie)
        elif distance <= self.vision and  distance>50:
            self.marcher_vers(proie)
        self.consommer_energie("chasser")

    def mordre(self, proie):

        if self.energie > 10:  # Vérifie si l'animal a assez d'énergie pour mordre
            
            proie.subir_degats(20)  
            self.energie -= 2  
        self.consommer_energie("mordre")

    def subir_degats(self, degats):
        """Subit des dégâts et réagit en conséquence."""
        self.poids -= degats  # Réduit la masse de l'animal (peut être lié à sa santé)
        if self.poids <= 0:
            print(f"{self.nom} est trop affaibli et meurt.")
            self.etat = "mort"  # L'animal est mort
        

    
    def se_reposer(self):
        """L'animal se repose pour récupérer de l'énergie."""
        if self.energie < 100:
            self.energie = min(100, self.energie + 10)  # Récupère 10 points d'énergie
            
    def mettre_a_jour_faim(self):
        """Met à jour le niveau de faim en fonction de la nourriture mangée."""
        # Si l'animal a mangé, la faim diminue
        self.faim -= self.nourriture_mangée * 10  
        
        if self.faim < 0:
            self.faim = 0

        if self.nourriture_mangée == 0:
            self.faim += 5  
        
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
        """Consomme de l'énergie en fonction du poids, de la vitesse, de l'activité et de la fatigue accumulée."""
        
        # Facteurs d'activité
        facteurs_activite = {
            "repos": 0.2,
            "marche": 1,
            "course": 2,
            "chasse": 3,
            "fuite": 3
        }
        
        # Récupération du facteur d'activité (défaut = marche)
        F_activite = facteurs_activite.get(activite, 1)
        
        # Métabolisme de base
        metabolism_base = 0.05 * self.poids
        
        # Coût énergétique lié à la vitesse
        cout_vitesse = (self.vitesse ** 1.5) / 5
        
        # Fatigue accumulée (augmente le coût si l'animal fait un effort prolongé)
        facteur_fatigue = 1 + 0.1 * self.fatigue
        
        # Calcul de la consommation
        consommation = (metabolism_base + cout_vitesse) * F_activite * facteur_fatigue * duree
        
        # Réduction de l’énergie
        self.energie = max(0, self.energie - consommation)
        
        # Augmentation de la fatigue
        if activite in ["course", "chasse", "fuite"]:
            self.fatigue += duree  # Courir ou chasser fatigue plus
        else:
            self.fatigue = max(0, self.fatigue - duree / 2)  # La fatigue diminue avec le temps

        # Vérification de l’épuisement
        if self.energie == 0:
            self.etat = "fatigué"
       
    def dormir(self):
        return
    def chercher_de_la_nourriture():
        return
    def se_cacher(self):
        return
    def regrouper(self):
        return
    def explorer(self):
        return
    def attribuer_couleur(self):
        couleurs = {
            "ours": "#8B4513",    # Marron
            "lion": "#FFD700",    # Or
            "gazelle": "#DAA520", # Doré
            "lapin": "#FFFFFF",   # Blanc
        }
        return couleurs.get(self.nom.lower(), "#808080")  # Gris par défaut si inconnu


class SystemeTerritorial:
    def __init__(self):
        self.noeuds = {}  # clé = espèce, valeur = {'position': (x, y), 'force': (fx, fy)}

    def centre_de_gravite(animaux, espece):
        positions = [(a.x, a.y) for a in animaux if a.espece == espece]
        if not positions:
            return (0, 0)
        x = sum(p[0] for p in positions) / len(positions)
        y = sum(p[1] for p in positions) / len(positions)
        return (x, y)


    def mettre_a_jour(self, animaux):
        # Mise à jour des positions (centre de gravité)
        for espece in set(a.nom for a in animaux):
            pos = self.centre_de_gravite(animaux, espece)
            self.noeuds.setdefault(espece, {'position': pos, 'force': (0, 0)})
            self.noeuds[espece]['position'] = pos

        # Appliquer des forces de répulsion entre territoires
        for espece1 in self.noeuds:
            fx, fy = 0, 0
            x1, y1 = self.noeuds[espece1]['position']
            for espece2 in self.noeuds:
                if espece1 == espece2:
                    continue
                x2, y2 = self.noeuds[espece2]['position']
                dx = x1 - x2
                dy = y1 - y2
                distance = (dx**2 + dy**2)**0.5 + 0.01
                force = 100 / distance  # force de répulsion
                fx += dx / distance * force
                fy += dy / distance * force
            self.noeuds[espece1]['force'] = (fx, fy)

    def appliquer_deplacement(self, facteur=0.1):
        for espece, data in self.noeuds.items():
            x, y = data['position']
            fx, fy = data['force']
            self.noeuds[espece]['position'] = (x + facteur * fx, y + facteur * fy)


#simulation = Simulation()

#Démarre la simulation manuellement
#asyncio.run(simulation.demarrer()) 