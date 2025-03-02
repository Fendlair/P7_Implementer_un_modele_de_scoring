# Client Loan Approval API

## Table des Matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Structure](#structure)

## Description

Ce projet est une API de prédiction de l'approbation de prêts clients. L'API utilise FastAPI pour fournir des prédictions basées sur un modèle XGBoost de machine learning. Le modèle est chargé à partir d'un fichier `.pkl`. Il est composé d'un scaler (RobustScaler) et du modèle XGBoost. Une app streamlit est disponible pour faire les prédictions. 

## Fonctionnalités

- Prédiction de l'approbation de prêts clients basée sur des caractéristiques spécifiques.
- Retourne la probabilité de remboursement du client.
- Tests unitaires pour vérifier les entrées et les réponses de l'API.
- Intégration continue avec GitHub Actions pour exécuter les tests automatiquement.

## Copier le projet

Clonez le dépôt :

   ```bash
   git clone https://github.com/Fendlair/P7_Implementer_un_modele_de_scoring
   ```

## Structure des fichiers

```
client_loan_approval/
├── app/
│   ├── init.py            # Fichier d'initialisation du package
│   ├── main.py            # Point d'entrée principal de l'API FastAPI
│   └── model.py           # Fichier contenant les fonctions de chargement du modèle
│
├── model/
│   └── model.pkl          # Fichier du modèle sérialisé (pipeline)
│
├── dashboard.py           # Fichier contenant le code du dashboard Streamlit
│
├── requirements.txt       # Fichier listant les dépendances Python du projet
│
└── README.md              # Fichier d'explication de l'organisation des fichiers (ce fichier)
```