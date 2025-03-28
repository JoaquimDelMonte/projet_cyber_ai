# Projet Cyber AI

## Descriptif du projet
Le projet a pour objectif de développer un outil basé sur l'intelligence artificielle pour la cybersécurité. Il vise à analyser des données, détecter des anomalies et prévenir les menaces. Ce projet est à but éducatif.

## Auteurs
- **Joaquim Del Monte**
- **Paul Dereeper**

## Instructions et exemples d'utilisation

### Utilisation :
- Traitement initial des datasets :
    python ./data_preprocessing.py
  
- Entrainement du modèle :
    python ./model.py

- Lancement de la capture live :
    choisir l'interface correspondante ligne 259
    exemple : **capture = pyshark.LiveCapture(interface='Wi-Fi')**
    python ./capture.py

- Lancement de l'IPS :
    python ./main.py
    - pour analyser un dataset labellisé choisir l'option 1 puis entrer le chemin d'accès du fichier .csv à analyser
    - pour lancer l'analyse live, lancer d'abord capture.py puis lancer main.py choisir l'option 2 puis en chemin d'accès rentrer : flows.csv 
