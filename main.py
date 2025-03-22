import pandas as pd
import numpy as np
import joblib
import time
import argparse
from sklearn.metrics import classification_report, confusion_matrix

def niveau_confiance(proba_max):
    """
    Détermine le niveau de confiance en fonction de la probabilité maximale.
    """
    if proba_max < 0.6:
        return 1  # Faible confiance
    elif proba_max < 0.85:
        return 2  # Confiance moyenne
    else:
        return 3  # Confiance élevée

def main():
    parser = argparse.ArgumentParser(
        description="Script de prédiction et calcul de statistiques détaillées par catégorie pour l'IDS"
    )
    parser.add_argument("input_csv", type=str, help="Chemin vers le fichier CSV contenant les données à prédire")
    args = parser.parse_args()
    
    # Chargement du modèle et du scaler préalablement enregistrés
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Chargement des données
    data = pd.read_csv(args.input_csv)
    
    # Vérifier que la colonne "Label" existe
    if "Label" not in data.columns:
        print("Erreur : Le fichier CSV ne contient pas de colonne 'Label'.")
        return
    else:
        # Conserver les vrais labels avant de supprimer la colonne
        true_labels = data["Label"].copy()
        features = data.drop("Label", axis=1)
    

    
    # Application de la normalisation Min-Max sur les données à prédire
    features_scaled = scaler.transform(features)
    
    # Initialisation des listes pour stocker les résultats
    predictions = []
    proba_max_list = []
    confiance_list = []
    
    print("Début des prédictions :\n")
    for i, row in enumerate(features_scaled):
        row = row.reshape(1, -1)
        pred = model.predict(row)[0]
        predictions.append(pred)
        proba = model.predict_proba(row)[0]
        max_proba = max(proba)
        proba_max_list.append(max_proba)
        conf = niveau_confiance(max_proba)
        confiance_list.append(conf)
        print(f"Instance {i+1} : Prédiction = {pred}, Proba max = {max_proba:.4f}, Niveau de confiance = {conf}")

    
    # Création du DataFrame de résultats
    result_df = pd.DataFrame({
        "Vraie Classe": true_labels,
        "Prédiction": predictions,
        "Proba max": proba_max_list,
        "Niveau de confiance": confiance_list
    })
    
    # Calcul d'une colonne indiquant si la prédiction est correcte (1) ou non (0)
    result_df["Correct"] = (result_df["Vraie Classe"] == result_df["Prédiction"]).astype(int)
    # Calcul de l'écart entre la prédiction et la vraie classe
    result_df["Écart"] = abs(result_df["Vraie Classe"] - result_df["Prédiction"])
    
    print("\nTableau des résultats :")
    print(result_df)
    
    # Calcul de statistiques détaillées par catégorie
    stats = result_df.groupby("Vraie Classe").agg(
        Nombre_d_instances=("Vraie Classe", "count"),
        Nombre_corrects=("Correct", "sum"),
        Taux_precision=("Correct", "mean"),
        Moyenne_proba_max=("Proba max", "mean"),
        Moyenne_niveau_confiance=("Niveau de confiance", "mean"),
        Erreur_moyenne=("Écart", "mean")
    ).reset_index()
    
    print("\nStatistiques détaillées par catégorie :")
    print(stats)
    
    # Affichage du rapport de classification et de la matrice de confusion
    print("\nRapport de classification :")
    print(classification_report(true_labels, predictions))
    
    print("\nMatrice de confusion :")
    print(confusion_matrix(true_labels, predictions))

if __name__ == "__main__":
    main()
