# Client Loan Approval API

## Table des Matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)

## Description

Ce projet est une API de prédiction de l'approbation de prêts clients. L'API utilise FastAPI pour fournir des prédictions basées sur un modèle XGBoost de machine learning. Le modèle est chargé à partir d'un fichier `.pkl`. Il est composé d'un scaler (RobustScaler) et du modèle XGBoost. Une app streamlit est disponible pour faire les prédictions. 

## Fonctionnalités

- Prédiction de l'approbation de prêts clients basée sur des caractéristiques spécifiques.
- Retourne la probabilité de remboursement du client.
- Tests unitaires pour vérifier les entrées et les réponses de l'API.
- Intégration continue avec GitHub Actions pour exécuter les tests automatiquement.

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/Fendlair/P7_Implementer_un_modele_de_scoring
