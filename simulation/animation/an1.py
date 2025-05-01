import pandas as pd

# 1. Charger le DataFrame
df = pd.read_csv("/workspaces/Projet_simulation/simulation/model_IA/donnees_entrainement.csv")

# 2. Calculer les pourcentages par décision
pourcentages_decisions = df['decision'].value_counts(normalize=True) * 100

# 3. Arrondir à 2 décimales et afficher
print("Pourcentages par décision :")
print(pourcentages_decisions.round(2).to_string())
