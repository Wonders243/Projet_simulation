from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import SignupForm, ConnexionForm
from django.contrib import messages

def signup(request):
    form = SignupForm(request.POST)
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()
            login(request, user)
            return redirect('accueil')  # Remplace 'home' par ta page d'accueil
        else:
            form = SignupForm()
    return render(request, 'inscription.html',{'form': form})

def connexion(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('accueil_simulation')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")

    form = ConnexionForm()
    return render(request, 'connexion.html', {'form': form})


def accueil(request):
    return render(request, 'index.html')

def deconnexion(request):
    logout(request)
    return redirect('accueil')  # Redirige vers la page de connexion après la déconnexion
