import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from animation import routing  # Importe les bons routing de ton application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'simulation.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns  # Assurez-vous que votre routing est bien utilisé
        )
    ),
})
