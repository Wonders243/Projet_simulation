from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm

from django import forms
from django.contrib.auth.models import User
class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'w-full p-2 border rounded-lg',
        'placeholder': 'Mot de passe'
    }))
    password_confirm = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'w-full p-2 border rounded-lg',
        'placeholder': 'Confirmez le mot de passe'
    }))

    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full p-2 border rounded-lg',
                'placeholder': 'Nom d’utilisateur'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full p-2 border rounded-lg',
                'placeholder': 'Email'
            }),
        }

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password and password_confirm and password != password_confirm:
            self.add_error('password_confirm', "Les mots de passe ne correspondent pas.")

        return cleaned_data


class ConnexionForm(AuthenticationForm):
    username = forms.CharField(
        label="Nom d'utilisateur",
        widget=forms.TextInput(attrs={
            'class': 'w-full p-2 border rounded bg-gray-700 text-white',
            'placeholder': 'Entrez votre nom d\'utilisateur'
        })
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full p-2 border rounded bg-gray-700 text-white',
            'placeholder': 'Entrez votre mot de passe'
        })
    )
