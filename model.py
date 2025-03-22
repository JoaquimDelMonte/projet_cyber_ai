import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Fixer une graine aléatoire pour la reproductibilité
RANDOM_STATE_SEED = 12


df_equal = pd.read_csv("dataset_combined.csv")
# Séparation des données en train (80%) et test (20%)
train, test = train_test_split(df_equal, test_size=0.2, random_state=RANDOM_STATE_SEED)



# Sélection des colonnes numériques pour le scaling
numerical_columns = [col for col in df_equal.columns if col not in ["Label"]]

# Appliquer la normalisation Min-Max
scaler = MinMaxScaler().fit(train[numerical_columns])
train[numerical_columns] = scaler.transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])
# Séparation des features et du label
y_train = np.array(train.pop("Label"))
X_train = train.values
y_test = np.array(test.pop("Label"))
X_test = test.values

# Initialisation du modèle de régression logistique multinomiale
log_reg = LogisticRegression(random_state=RANDOM_STATE_SEED, max_iter=1000, multi_class="multinomial", solver="lbfgs")

# Entraînement du modèle
log_reg.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Probabilités associées aux prédictions
y_probs = log_reg.predict_proba(X_test)

# Définition des niveaux de confiance en fonction des probabilités
def niveau_confiance(proba_max):
    if proba_max < 0.6:
        return 1  # 
    elif proba_max < 0.85:
        return 2  # Moyen
    else:
        return 3  # Élevé

# Associer chaque prédiction à un niveau de confiance
niveaux_risque = [niveau_confiance(max(p)) for p in y_probs]

# Résumé des résultats avec niveaux de confiance
result_df = pd.DataFrame({
    "Vraie Classe": y_test,
    "Prédiction": y_pred,
    "Proba max": [max(p) for p in y_probs],
    "Niveau de confiance": niveaux_risque
})

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n Accuracy: {accuracy:.4f}")
print(result_df)


import joblib

# Sauvegarde du modèle et du scaler
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
