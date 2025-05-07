# 🌿 Projet de Simulation d'Écosystème — L3 S6

Projet réalisé dans le cadre du semestre 6 de la Licence Mathematique, Mecanique, Informatique à l’Université de Nîmes.

## 📌 Objectif du projet

Ce projet vise à simuler un écosystème dynamique dans lequel interagissent différents types d'animaux et de ressources naturelles (plantes, eau). Chaque animal possède des caractéristiques propres (vitesse, vision, énergie, faim, etc.) et prend des décisions comportementales basées sur un modèle LSTM de prédiction.

## 🧠 Fonctionnalités principales

- **Déplacement aléatoire, chasse, fuite, et exploration** selon l’état interne des animaux.
- **Prise de décision intelligente** via un modèle LSTM entraîné sur des données comportementales simulées.
- **Consommation de ressources naturelles** : les plantes pour les herbivores, les proies pour les carnivores.
- **Vision conique dynamique** utilisant un champ directionnel basé sur `dx`, `dy`.
- **Évolution de l’état interne** : énergie, faim, soif, fatigue influencent les décisions.
- **Mécanique de reproduction et de mortalité**.

## 🧬 Technologies utilisées

- Python 3.11
- Django + Channels (WebSocket) pour la gestion en temps réel
- NumPy, Math, Random
- TensorFlow / Keras pour le modèle LSTM
- HTML/CSS/JS (pour l’interface, si utilisée)

## 🚀 Lancement de la simulation

1. Cloner le projet :
   ```bash
   git clone https://github.com/ton-utilisateur/Projet_simulation.git
   cd Projet_simulation

