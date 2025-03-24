from django.urls import re_path
from animation.Class_moteur import Simulation  # Assurez-vous d'avoir un consumer WebSocket configuré

websocket_urlpatterns = [
    re_path(r'ws/animals/', Simulation.as_asgi()),  # Le chemin de votre WebSocket
]
