# Analyse automatique de la qualité d'un jeu de données animaux
import pandas as pd

def analyser_qualite_donnees(df):
    rapport = {}

    # Nombre de lignes / colonnes
    rapport['dimensions'] = df.shape

    # Valeurs manquantes
    rapport['valeurs_manquantes'] = df.isnull().sum().to_dict()

    # Colonnes constantes
    constantes = df.loc[:, df.nunique() <= 1].columns.tolist()
    rapport['colonnes_constantes'] = constantes

    # Analyse par type
    rapport['colonnes_categorielles'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    rapport['colonnes_numeriques'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Statistiques descriptives pour les colonnes numériques
    rapport['statistiques_numeriques'] = df.describe().T.to_dict()

    # Distributions de valeurs pour les catégorielles
    rapport['distributions_categorielles'] = {
        col: df[col].value_counts().to_dict()
        for col in rapport['colonnes_categorielles']
    }

    # Colonnes binaires
    rapport['colonnes_binaires'] = [col for col in df.columns if df[col].dropna().nunique() == 2]

    return rapport

# Exemple d'utilisation :
df = pd.read_csv("/workspaces/Projet_simulation/simulation/model_IA/donnees_entrainement.csv")
rapport = analyser_qualite_donnees(df)
import pprint; pprint.pprint(rapport)
