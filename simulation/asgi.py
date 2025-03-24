import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from animation import routing  # Assurez-vous que le routing est bien importé depuis ton app

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'simulation.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns  # La route WebSocket de ton application
        )
    ),
})
