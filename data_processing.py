import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from datasets import load_dataset


##### FONCTIONS UTILITAIRES #####

def generate_random_ip():
    """Génère une adresse IPv4 aléatoire."""
    return ".".join(str(np.random.randint(1, 255)) for _ in range(4))



#################################



# Charger les deux fichiers individuellement

dataset1 = load_dataset("pauldereep/projet_cyber_ai", data_files="DDoS_dataset.csv")["train"]
dataset2 = load_dataset("pauldereep/projet_cyber_ai", data_files="02-14-2018_processed.csv")["train"]

# Convertir en DataFrame pandas
df1 = dataset1.to_pandas()
df2 = dataset2.to_pandas()

df_combined = pd.concat([df1,df2])

# Nettoyer les noms de colonnes (supprimer les espaces en début et fin)
df_combined.columns = df_combined.columns.str.strip()

df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.dropna(inplace=True)
df_combined = df_combined.sample(frac=1, random_state=42)
# Échantillonner n_elements exemples pour chaque label
df_combined["Label"] = df_combined["Label"].replace("BENIGN", "Benign")
df_benign = df_combined[df_combined["Label"] == "Benign"][:15000]
df_ddos = df_combined[df_combined["Label"] == "DDoS"][:15000]
df_ftp = df_combined[df_combined["Label"] == "FTP-BruteForce"][:15000]
df_ssh = df_combined[df_combined["Label"] == "SSH-Bruteforce"][:15000]
print(len(df_benign))
print(len(df_ddos))
print(len(df_ftp))
print(len(df_ssh))
print(df_combined["Label"].unique())

# Concaténer les deux classes pour obtenir un dataset équilibré
df_balanced = pd.concat([df_benign, df_ddos, df_ftp, df_ssh], axis=0).reset_index(drop=True)

# Remplacer les labels textuels par des valeurs numériques (0 pour BENIGN, 1 pour DDoS)
df_balanced.replace({"Benign": 0, "DDoS": 1, "FTP-BruteForce" : 2, "SSH-Bruteforce" : 3}, inplace=True)

# Génération d'une colonne d'IP aléatoires
df_balanced["Source IP"] = [generate_random_ip() for _ in range(len(df_balanced))]

# Déplacer la colonne "IP" en première position
cols = ["Source IP"] + [col for col in df_balanced.columns if col != "Source IP"]
df_balanced = df_balanced[cols]

# Sauvegarde
df_balanced.to_csv("dataset_new_balanced.csv", index=False)
print("✅ Traitement des données terminé ! Dataset sauvegardé sous 'dataset_new_balanced.csv'")

# Sauvegarder le dataset prétraité et équilibré
df_balanced.to_csv("dataset_new_balanced.csv", index=False)
print("Traitement des données terminé!")

