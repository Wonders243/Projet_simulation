from django.urls import re_path
from animation import Class_moteur  # Assurez-vous d'avoir un consumer WebSocket configuré

websocket_urlpatterns = [
    re_path(r'ws/animals/', Class_moteur.Simulation.as_asgi()),  # Le chemin de votre WebSocket
]
