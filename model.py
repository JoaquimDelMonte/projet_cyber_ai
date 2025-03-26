import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Fixer une graine aléatoire pour la reproductibilité
RANDOM_STATE_SEED = 12

# Charger le nouveau dataset prétraité et équilibré
df = pd.read_csv("dataset_new_balanced.csv")

# Nettoyer les noms de colonnes (supprimer les espaces inutiles)
df.columns = df.columns.str.strip()

# Séparation des données en ensemble d'entraînement (80%) et de test (20%)
train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE_SEED)

# Sélectionner toutes les colonnes numériques (en excluant la colonne "Label")
numerical_columns = [col for col in train.columns if col not in ["Label", "Source IP"]]

# Appliquer la normalisation Min-Max sur les données d'entraînement et de test
scaler = MinMaxScaler().fit(train[numerical_columns])
train[numerical_columns] = scaler.transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])

# Séparation des IPs
ip_train = train.pop("Source IP").values
ip_test = test.pop("Source IP").values

# Séparation des labels
y_train = train.pop("Label").values
y_test = test.pop("Label").values

# Features
X_train = train.values
X_test = test.values

# Définition des poids de classes pour favoriser le recall de Benign (classe 0)
# et améliorer la précision pour DDoS (classe 1).
# Ici, on augmente le poids de la classe 0 (par exemple 2) et on baisse celui de la classe 1 (par exemple 0.5).
# Les classes 2 et 3 gardent le poids par défaut (1).
class_weights = {0: 1.8, 1: 0.6, 2: 1, 3: 1}

# Initialisation et entraînement du modèle de régression logistique avec les poids de classes
log_reg = LogisticRegression(random_state=RANDOM_STATE_SEED, max_iter=1000, class_weight=class_weights)
log_reg.fit(X_train, y_train)

# Prédictions sur l'ensemble de test et récupération des probabilités associées
y_pred = log_reg.predict(X_test)
y_probs = log_reg.predict_proba(X_test)

# Définition d'une fonction pour déterminer le niveau de confiance à partir de la probabilité maximale
def niveau_confiance(proba_max):
    if proba_max < 0.6:
        return 1  # Faible confiance
    elif proba_max < 0.85:
        return 2  # Confiance moyenne
    else:
        return 3  # Forte confiance

# Associer à chaque prédiction son niveau de confiance
niveaux_confiance = [niveau_confiance(np.max(prob)) for prob in y_probs]

# Création d'un DataFrame récapitulatif des résultats
result_df = pd.DataFrame({
    "Source IP": ip_test,
    "Vraie Classe": y_test,
    "Prédiction": y_pred,
    "Proba max": [np.max(prob) for prob in y_probs],
    "Niveau de confiance": niveaux_confiance
})

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print(class_report)
print(result_df.head())

# Visualisation 1 : Heatmap de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["BENIGN", "DDoS", "FTP-BruteForce", "SSH-Bruteforce"],
            yticklabels=["BENIGN", "DDoS", "FTP-BruteForce", "SSH-Bruteforce"])
plt.xlabel("Prédiction")
plt.ylabel("Vraie Classe")
plt.title("Matrice de Confusion")
plt.show()

# Visualisation 2 : Répartition des niveaux de confiance
plt.figure(figsize=(6, 4))
sns.countplot(x="Niveau de confiance", data=result_df, palette="Set2")
plt.title("Répartition des Niveaux de Confiance")
plt.xlabel("Niveau de Confiance")
plt.ylabel("Nombre d'exemples")
plt.show()

# Sauvegarde du modèle et du scaler
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
