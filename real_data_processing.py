import pandas as pd
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Compter le nombre de colonnes (après suppression des 4 premières colonnes) et le nombre de lignes par label dans un CSV labellisé"
    )
    parser.add_argument("input_csv", type=str, help="Chemin vers le fichier CSV labellisé")
    args = parser.parse_args()
    
    # Chargement des données
    df = pd.read_csv(args.input_csv)
    
    # Suppression des 4 premières colonnes
    df = df.drop(df.columns[:4], axis=1)
    df.replace(to_replace="Benign", value=0, inplace=True)
    df.replace(to_replace="Bot", value=9, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        df["Timestamp"] = df["Timestamp"].astype(int) / 10**9  # Convertir en secondes
  
    df.to_csv("dataset_test.csv", index=False)
    # Comptage des colonnes après suppression
    n_cols = len(df.columns)
    print(f"Nombre de colonnes après suppression des 4 premières colonnes : {n_cols}")
    
    # Vérification de la présence de la colonne "Label"
    if "Label" not in df.columns:
        print("Erreur : Le CSV ne contient pas de colonne 'Label'.")
        return
    
    # Comptage du nombre de lignes par label
    label_counts = df["Label"].value_counts()
    print("\nNombre de lignes par label :")
    for label, count in label_counts.items():
        print(f"  {label} : {count}")

if __name__ == "__main__":
    main()
