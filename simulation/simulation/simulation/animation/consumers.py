import json
import random
import asyncio
import math
from channels.generic.websocket import AsyncWebsocketConsumer


class AnimalConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("Connexion WebSocket établie !")

        # Liste d'animaux, chaque animal peut être un lion (prédateur) ou une gazelle (proie)
        self.animals = self.create_animals()

        asyncio.create_task(self.update_animals())

    def create_animals(self):
        animals = []
        for _ in range(11):  # Ajoute 2 lions (prédateurs)
            lion = {
                "id": "lion",
                "x": random.randint(100, 700),
                "y": random.randint(100, 500),
                "dx": random.uniform(-1, 1),
                "dy": random.uniform(-1, 1),
                "speed": 2,
                "max_speed": 20,
                "acceleration": 5,
                "braking": 3,
                "vision": 100,
                "view_angle": math.radians(60),
                "angle": random.uniform(0, 2 * math.pi),
                "chasing": False,
                "color": "red",
                "type": "predator"  # Spécifie que c'est un prédateur
            }
            animals.append(lion)

        for _ in range(33):  # Ajoute 5 gazelles (proies)
            gazelle = {
                "id": "gazelle",
                "x": random.randint(100, 700),
                "y": random.randint(100, 500),
                "dx": random.uniform(-1, 1),
                "dy": random.uniform(-1, 1),
                "speed": 2,
                "max_speed": 15,
                "acceleration": 7,
                "braking": 6,
                "vision": 75,
                "view_angle": math.radians(90),
                "angle": random.uniform(0, 2 * math.pi),
                "color": "green",
                "type": "prey"  # Spécifie que c'est une proie
            }
            animals.append(gazelle)

        return animals

    async def update_animals(self):
        while True:
            for animal in self.animals:
                if animal["type"] == "predator":
                    self.update_lion(animal)
                elif animal["type"] == "prey":
                    self.update_gazelle(animal)

            await self.send(text_data=json.dumps(self.animals))
            await asyncio.sleep(0.05)

    def distance(self, a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    def angle_between(self, a, b):
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        return math.atan2(dy, dx)

    def in_vision(self, a, b):
        # Vérifie si l'animal b est dans le champ de vision de l'animal a
        angle_to_b = self.angle_between(a, b)
        angle_diff = abs(a["angle"] - angle_to_b)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        distance_to_b = self.distance(a, b)
        return distance_to_b <= a["vision"] and angle_diff <= a["view_angle"] / 2

    def update_lion(self, lion):
        # Recherche des proies (gazelles)
        prey_in_sight = [animal for animal in self.animals if animal["type"] == "prey" and self.in_vision(lion, animal)]

        if prey_in_sight:
            nearest_prey = min(prey_in_sight, key=lambda animal: self.distance(lion, animal))

            # Le lion chasse la gazelle
            if not lion["chasing"]:
                lion["chasing"] = True

            # Le lion tourne pour regarder la gazelle
            angle_to_prey = self.angle_between(lion, nearest_prey)
            lion["angle"] = angle_to_prey

            # Déplace le lion vers la gazelle
            dx = nearest_prey["x"] - lion["x"]
            dy = nearest_prey["y"] - lion["y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist > 2:
                # Accélération vers la gazelle
                lion["dx"] += (dx / dist) * lion["acceleration"]
                lion["dy"] += (dy / dist) * lion["acceleration"]

            # Empêche le lion de devenir immobile en dessous d'une certaine vitesse
            speed = math.sqrt(lion["dx"] ** 2 + lion["dy"] ** 2)
            if speed < 0.1:  # Vitesse minimale pour que le lion continue à bouger
                lion["dx"] = lion["dy"] = 0  # Empêche d'aller à zéro
            elif speed > lion["max_speed"]:
                lion["dx"] *= lion["max_speed"] / speed
                lion["dy"] *= lion["max_speed"] / speed
        else:
            lion["chasing"] = False
            # Si aucune proie n'est détectée, le lion arrête de chasser et ralentit
            lion["dx"] *= 0.5  # Ralentit progressivement
            lion["dy"] *= 0.5  # Ralentit progressivement

            # Limiter la vitesse du lion
            speed = math.sqrt(lion["dx"] ** 2 + lion["dy"] ** 2)
            if speed > lion["max_speed"]:
                lion["dx"] *= lion["max_speed"] / speed
                lion["dy"] *= lion["max_speed"] / speed

        # Déplacement du lion
        lion["x"] += lion["dx"]
        lion["y"] += lion["dy"]

    def update_gazelle(self, gazelle):
        # Vérifie si un lion est dans le champ de vision de la gazelle
        predator_in_sight = [animal for animal in self.animals if animal["type"] == "predator" and self.in_vision(gazelle, animal)]

        if predator_in_sight:
            # La gazelle fuit le lion
            nearest_predator = min(predator_in_sight, key=lambda animal: self.distance(gazelle, animal))

            # La gazelle fuit dans la direction opposée
            angle_to_predator = self.angle_between(gazelle, nearest_predator)
            gazelle["angle"] = angle_to_predator + math.pi  # Regard opposé à celui du lion

            dx = gazelle["x"] - nearest_predator["x"]
            dy = gazelle["y"] - nearest_predator["y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist > 1:
                # Accélération dans la direction opposée au lion
                gazelle["dx"] += (dx / dist) * gazelle["acceleration"]
                gazelle["dy"] += (dy / dist) * gazelle["acceleration"]

            # Limiter la vitesse de la gazelle
            speed = math.sqrt(gazelle["dx"] ** 2 + gazelle["dy"] ** 2)
            if speed > gazelle["max_speed"]:
                gazelle["dx"] *= gazelle["max_speed"] / speed
                gazelle["dy"] *= gazelle["max_speed"] / speed

            # Feinte si le lion est trop proche
            if dist < 20:
                gazelle["dx"] *= -0.5
                gazelle["dy"] *= -0.5
        else:
            # Si aucun lion n'est détecté, la gazelle se déplace de manière aléatoire
            gazelle["dx"] += random.uniform(-0.5, 0.5)
            gazelle["dy"] += random.uniform(-0.5, 0.5)

        # Déplacement de la gazelle
        gazelle["x"] += gazelle["dx"]
        gazelle["y"] += gazelle["dy"]

