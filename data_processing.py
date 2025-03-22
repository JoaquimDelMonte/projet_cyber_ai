import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Charger les dataset
df_dataset1 = pd.read_csv("./dataset/02-14-2018.csv")
df_dataset2 = pd.read_csv("./dataset/02-15-2018.csv")
df_dataset3 = pd.read_csv("./dataset/02-16-2018.csv")
df_dataset5 = pd.read_csv("./dataset/02-21-2018.csv")
df_dataset10 = pd.read_csv("./dataset/03-02-2018.csv")
# Fusionner avec l'ancien dataset
df_combined = pd.concat([df_dataset1, df_dataset2, df_dataset3, df_dataset5, df_dataset10], axis=0)
# Équilibrer le dataset (même nombre d'exemples pour chaque classe)
df1 = df_combined[df_combined["Label"] == "Benign"][:32000]
df2 = df_combined[(df_combined["Label"] == "FTP-BruteForce")][:10000]
df3 = df_combined[df_combined["Label"] == "SSH-Bruteforce"][:10000]
df4 = df_combined[df_combined["Label"] == "DoS attacks-GoldenEye"][:10000]
df5 = df_combined[df_combined["Label"] == "DoS attacks-Slowloris"][:10000]
df6 = df_combined[df_combined["Label"] == "DoS attacks-SlowHTTPTest"][:10000]
df7 = df_combined[df_combined["Label"] == "DoS attacks-Hulk"][:10000]
df8 = df_combined[df_combined["Label"] == "DDoS attacks-LOIC-HTTP"][:10000]
df9 = df_combined[df_combined["Label"] == "DDOS attack-HOIC"][:10000]
df10 = df_combined[df_combined["Label"] == "Bot"][:10000]
df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)
df3 = df3.reset_index(drop=True)
df4 = df4.reset_index(drop=True)
df5 = df5.reset_index(drop=True)
df6 = df6.reset_index(drop=True)
df7 = df7.reset_index(drop=True)
df8 = df8.reset_index(drop=True)
df9 = df9.reset_index(drop=True)
df10 = df10.reset_index(drop=True)
df_equal = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=0)
# Remplacement des labels par des valeurs numériques
df_equal.replace(to_replace="Benign", value=0, inplace=True)
df_equal.replace(to_replace="SSH-Bruteforce", value=1, inplace=True)
df_equal.replace(to_replace="FTP-BruteForce", value=2, inplace=True)
df_equal.replace(to_replace="DoS attacks-GoldenEye", value=3, inplace=True)
df_equal.replace(to_replace="DoS attacks-Slowloris", value=4, inplace=True)
df_equal.replace(to_replace="DoS attacks-SlowHTTPTest", value=5, inplace=True)
df_equal.replace(to_replace="DoS attacks-Hulk", value=6, inplace=True)
df_equal.replace(to_replace="DDoS attacks-LOIC-HTTP", value=7, inplace=True)
df_equal.replace(to_replace="DDOS attack-HOIC", value=8, inplace=True)
df_equal.replace(to_replace="Bot", value=9, inplace=True)
# Vérifier si la colonne Timestamp existe et la convertir en format numérique
if "Timestamp" in df_equal.columns:
    df_equal["Timestamp"] = pd.to_datetime(df_equal["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    df_equal["Timestamp"] = df_equal["Timestamp"].astype(int) / 10**9  # Convertir en secondes
    df_equal.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_equal.dropna(inplace=True)
# Sauvegarder le dataset fusionné
df_equal.to_csv("dataset_combined.csv", index=False)
print("Data processing done!")