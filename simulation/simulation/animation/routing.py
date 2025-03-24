from django.urls import re_path
from animation import consumers  # Assurez-vous d'avoir un consumer WebSocket configuré

websocket_urlpatterns = [
    re_path(r'ws/animals/', consumers.AnimalConsumer.as_asgi()),  # Le chemin de votre WebSocket
]
