import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=InconsistentVersionWarning)


import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def ascii_welcome():
    art = r"""
  ____  _             _        _ _           _   
 |  _ \| |           | |      (_) |         | |  
 | |_) | | __ _ _ __ | | __    _| |__   ___ | |_ 
 |  _ <| |/ _` | '_ \| |/ /   | | '_ \ / _ \| __|
 | |_) | | (_| | | | |   <    | | |_) | (_) | |_ 
 |____/|_|\__,_|_| |_|_|\_\   |_|_.__/ \___/ \__|
    """
    print(art)
    print("Bienvenue dans le système de prédiction IDS !\n")

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
    # Affichage du message d'accueil
    ascii_welcome()
    
    # Chargement du modèle et du scaler préalablement enregistrés
    try:
        model = joblib.load("logistic_regression_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du scaler : {e}")
        sys.exit(1)
    
    # Demande interactive du chemin vers le fichier CSV à prédire
    input_csv = input("Veuillez entrer le chemin vers le fichier CSV contenant les données à prédire : ")
    try:
        data = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {e}")
        sys.exit(1)
    
    # Vérifier que la colonne "Label" existe
    if "Label" not in data.columns:
        print("Erreur : Le fichier CSV ne contient pas de colonne 'Label'.")
        sys.exit(1)
    else:
        # Conserver les vrais labels avant de supprimer la colonne
        true_labels = data["Label"].copy()
        features = data.drop("Label", axis=1)
    
    # Application de la normalisation Min-Max sur les données à prédire
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        print(f"Erreur lors de la transformation des features : {e}")
        sys.exit(1)
    
    # Initialisation des listes pour stocker les résultats
    predictions = []
    proba_max_list = []
    confiance_list = []
    
    print("\nDébut des prédictions :\n")
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
    
    # Conversion des labels vrais en valeurs numériques à l'aide d'un mapping
    # Adaptez ce mapping en fonction de vos classes
    label_mapping = {"Benign": 0, "DDoS": 1, "FTP-BruteForce": 2, "SSH-Bruteforce": 3}
    result_df["Vraie Classe Num"] = result_df["Vraie Classe"].replace(label_mapping).astype(int)
    
    # Calcul de l'écart entre la prédiction et la vraie classe (sur des valeurs numériques)
    result_df["Écart"] = abs(result_df["Vraie Classe Num"] - result_df["Prédiction"])
    
    print("\nTableau des résultats :")
    print(result_df)
    
    # Calcul de statistiques détaillées par catégorie
    stats = result_df.groupby("Vraie Classe").agg(
        Nombre_d_instances=("Vraie Classe", "count"),
        Nombre_corrects=("Écart", lambda x: (x==0).sum()),
        Taux_precision=("Écart", lambda x: (x==0).mean()),
        Moyenne_proba_max=("Proba max", "mean"),
        Moyenne_niveau_confiance=("Niveau de confiance", "mean"),
        Erreur_moyenne=("Écart", "mean")
    ).reset_index()
    
    print("\nStatistiques détaillées par catégorie :")
    print(stats)
    
    # Affichage du rapport de classification et de la matrice de confusion (texte)
    print("\nRapport de classification :")
    print(classification_report(result_df["Vraie Classe Num"], predictions))
    
    cm = confusion_matrix(result_df["Vraie Classe Num"], predictions, labels=[0, 1, 2, 3])
    print("\nMatrice de confusion :")
    print(cm)
    
    # ---------------------------
    # Visualisations avec matplotlib
    # ---------------------------
    
    # 1. Matrice de confusion sous forme de heatmap
    all_labels = ["Benign", "DDoS", "FTP-BruteForce", "SSH-Bruteforce"]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=all_labels,
           yticklabels=all_labels,
           xlabel="Classe prédite",
           ylabel="Vraie classe",
           title="Matrice de Confusion")
    
    # Annoter chaque cellule de la matrice
    fmt = "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()
    
    # 2. Répartition des niveaux de confiance (diagramme en barres)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    niveau_counts = result_df["Niveau de confiance"].value_counts().sort_index()
    ax2.bar(niveau_counts.index.astype(str), niveau_counts.values, color='skyblue')
    ax2.set_xlabel("Niveau de Confiance")
    ax2.set_ylabel("Nombre d'exemples")
    ax2.set_title("Répartition des Niveaux de Confiance")
    plt.tight_layout()
    plt.show()
    
    # 3. Taux de précision par catégorie (diagramme en barres)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    # Calculer le taux de précision en pourcentage
    taux_precision = stats["Taux_precision"] * 100
    ax3.bar(stats["Vraie Classe"], taux_precision, color='green')
    ax3.set_xlabel("Classe (Vraie)")
    ax3.set_ylabel("Taux de précision (%)")
    ax3.set_title("Précision par catégorie")
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(all_labels)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
