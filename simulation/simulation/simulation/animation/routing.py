from django.urls import path
from .consumers import AnimalConsumer

websocket_urlpatterns = [
    path("ws/animals/", AnimalConsumer.as_asgi()),
]