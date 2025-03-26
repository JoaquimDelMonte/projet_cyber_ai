import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=InconsistentVersionWarning)

import pandas as pd
import numpy as np
import joblib
import sys
import time
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
    ascii_welcome()
    
    # Chargement du modèle et du scaler préalablement enregistrés
    try:
        model = joblib.load("logistic_regression_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du scaler : {e}")
        sys.exit(1)
    
    print("Choisissez le mode d'utilisation :")
    print("1 : Analyse d'un fichier statique (avec label) pour tester le modèle.")
    print("2 : Analyse live (d'un fichier qui est mis à jour en continu, sans comparaison avec le label).")
    mode = input("Entrez 1 ou 2 : ").strip()
    
    if mode == "1":
        # ---------------------------
        # Mode Test : Fichier statique avec label
        # ---------------------------
        file_path = input("Veuillez entrer le chemin du fichier CSV de test : ")
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier CSV : {e}")
            sys.exit(1)
        
        if "Label" not in data.columns:
            print("Erreur : Le fichier CSV ne contient pas de colonne 'Label'.")
            sys.exit(1)
        ips = data["Source IP"].copy()
        data.drop("Source IP", axis=1, inplace=True)
        true_labels = data["Label"].copy()
        features = data.drop("Label", axis=1)
        try:
            features_scaled = scaler.transform(features)
        except Exception as e:
            print(f"Erreur lors de la transformation des features : {e}")
            sys.exit(1)
        
        predictions = model.predict(features_scaled)
        proba_list = model.predict_proba(features_scaled)
        results = []
        for i in range(len(features_scaled)):
            max_proba = max(proba_list[i])
            conf = niveau_confiance(max_proba)
            print(f"Instance {i+1} : Prédiction = {predictions[i]}, Proba max = {max_proba:.4f}, Niveau de confiance = {conf}")
            results.append({
                "Vraie Classe": true_labels.iloc[i],
                "Prédiction": predictions[i],
                "Proba max": max_proba,
                "Niveau de confiance": conf,
                "Source IP": ips.iloc[i]
            })
        
        result_df = pd.DataFrame(results)
        # Ici on effectue la conversion en valeurs numériques via un mapping
        label_mapping = {"Benign": 0, "DDoS": 1, "FTP-BruteForce": 2, "SSH-Bruteforce": 3}
        result_df["Vraie Classe Num"] = result_df["Vraie Classe"].map(label_mapping).fillna(-1).astype(int)
        
        print("\nTableau des résultats :")
        print(result_df)
        
        print("\nRapport de classification :")
        print(classification_report(result_df["Vraie Classe Num"], predictions))
        
        cm = confusion_matrix(result_df["Vraie Classe Num"], predictions, labels=[0, 1, 2, 3])
        print("\nMatrice de confusion :")
        print(cm)
    
    elif mode == "2":
        # ---------------------------
        # Mode Live : Fichier mis à jour en continu (sans comparaison avec le label)
        # ---------------------------
        file_path = input("Veuillez entrer le chemin du fichier CSV live : ")
        last_processed = 0
        print("\nDébut de l'analyse live. Le programme surveillera le fichier et traitera les nouvelles lignes au fur et à mesure.")
        
        while True:
            try:
                data = pd.read_csv(file_path)
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data.dropna(inplace=True)
                ips = data["Source IP"].copy()
                data.drop("Source IP", axis=1, inplace=True)
            except Exception as e:
                print(f"Erreur lors du chargement du fichier CSV : {e}")
                time.sleep(2)
                continue
            
            # Si la colonne "Label" existe, on l'ignore
            if "Label" in data.columns:
                features = data.drop("Label", axis=1)
            else:
                features = data
            
            if data.shape[0] > last_processed:
                new_data = features.iloc[last_processed:]
                try:
                    features_scaled = scaler.transform(new_data)
                except Exception as e:
                    print(f"Erreur lors de la transformation des features : {e}")
                    time.sleep(2)
                    continue
                
                predictions = model.predict(features_scaled)
                proba_list = model.predict_proba(features_scaled)
                
                for i in range(new_data.shape[0]):
                    max_proba = max(proba_list[i])
                    conf = niveau_confiance(max_proba)
                    instance_index = last_processed + i + 1
                    list = ["Benign", "DDoS", "FTP-BruteForce", "SSH-Bruteforce"]
                    print(f"Instance {instance_index} : Prédiction = {list[predictions[i]]}, Proba max = {max_proba:.4f}, Niveau de confiance = {conf}, IP = {ips.iloc[instance_index-1]}")
                
                last_processed = data.shape[0]
            # Pause avant la prochaine vérification
            time.sleep(2)
    
    else:
        print("Option invalide. Veuillez relancer le programme et choisir 1 ou 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
