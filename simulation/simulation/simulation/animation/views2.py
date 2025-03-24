from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, "animations.html")

def accueil_simulation(request):
    return render(request, "accueil_simulation.html")