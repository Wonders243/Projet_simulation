import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from animation.routing import websocket_urlpatterns  # Remplace "myapp" par le nom de ton app

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'simulation.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})