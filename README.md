# AutoML Challenge – Méthodologie IA & Méthodes Classiques

## Objectif du projet

Ce projet a pour but de concevoir une **pipeline AutoML (Automated Machine Learning)** capable d'automatiser l’ensemble du processus d’apprentissage machine :

- Préparation et nettoyage des données  
- Détection automatique du type de tâche (classification ou régression)  
- Entraînement de plusieurs modèles issus de `scikit-learn`  
- Optimisation automatique des hyperparamètres  
- Évaluation selon les métriques les plus pertinentes  
- Génération automatique des rapports de résultats  

Ce travail est réalisé dans le cadre du module **Méthodologie pour l’IA** (M1 IA) et du module **Méthodes Classiques**, en utilisant le **cluster de calcul Skinner**.

---

## Données

Les données du challenge sont disponibles sur le cluster :

/info/corpus/ChallengeMachineLearning


Chaque dataset se compose de trois fichiers :

| Fichier | Contenu | Exemple |
|----------|----------|----------|
| `data.data` | Données d’entrée sans en-têtes | 45 32 1 0 ... |
| `data.solution` | Variables cibles (target) | 1 / 0 / valeurs continues |
| `data.type` | Type de chaque colonne (`Numerical` ou `Categorical`) | `Categorical`, `Numerical`, ... |

---

## Architecture du projet

L’arborescence complète du projet est la suivante :

automl_project/
├── pyproject.toml # Configuration du package et dépendances
├── README.md # Documentation du projet (ce fichier)
├── LICENSE # Licence du projet
├── docs/ # Documents de rendu
│ ├── paper/ # Article scientifique (Méthodologie IA)
│ └── slides/ # Présentation orale (Méthodes Classiques)
├── notebooks/ # Notebooks d’exploration (EDA, tests)
│ └── eda.ipynb
├── src/
│ └── automl/
│ ├── init.py # Initialisation du package
│ ├── cli.py # Interface en ligne de commande (optionnelle)
│ ├── config.py # Gestion des fichiers de configuration YAML
│ ├── data_loader.py # Lecture des fichiers .data / .solution / .type
│ ├── preprocess.py # Construction du préprocesseur (encodage, imputation)
│ ├── task_detector.py # Détection automatique du type de tâche
│ ├── models.py # Liste de modèles sklearn et grilles d’hyperparamètres
│ ├── tuner.py # Optimisation des hyperparamètres (GridSearch / Optuna)
│ ├── runner.py # Classe principale AutoML (fit, eval, save)
│ ├── metrics.py # Calcul des métriques selon le type de tâche
│ ├── utils.py # Fonctions utilitaires (logging, seed, I/O)
│ └── persistence.py # Sauvegarde / chargement des modèles (joblib)
├── experiments/ # Dossiers contenant les runs d’expérimentation
│ └── run_YYYYMMDD_HHMMSS/
│ ├── config.yaml
│ ├── results.json
│ ├── model.joblib
│ ├── metrics.csv
│ └── log.txt
├── scripts/
│ ├── run_local.sh # Script d’exécution locale
│ └── run_sbatch.sh # Script SLURM pour le cluster Skinner
├── tests/
│ └── test_data_loader.py # Tests unitaires
└── examples/
└── example_usage.py # Exemple minimal d’utilisation



## Installation

### 1. Cloner le projet

git clone https://github.com/username/automl_project.git
cd automl_project
2. Créer un environnement Python
Sur le cluster Skinner (ou en local) :


module load python/3.10  # ou utiliser conda
python -m venv venv
source venv/bin/activate

3. Installer les dépendances
pip install -e .
ou avec pyproject.toml :


pip install .
# Utilisation
Exemple simple depuis Python:
from automl.runner import AutoML

# Chemin de base des fichiers (sans extension)
data_path = "/info/corpus/ChallengeMachineLearning/data"

automl = AutoML(data_path)
automl.fit()
automl.eval()
Exemple en ligne de commande: 
python -m automl.runner --config configs/default.yaml
Exemple sur Skinner (via SLURM)
Script scripts/run_sbatch.sh :


#!/bin/bash
#SBATCH --job-name=automl_run
#SBATCH --output=automl_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.10
source ~/venv_automl/bin/activate

python -m automl.runner --config /home/s2508075/configs/config.yaml
Soumission :

sbatch scripts/run_sbatch.sh

# Configuration (config.yaml)

data_prefix: /info/corpus/ChallengeMachineLearning/data

split:
  test_size: 0.2
  val_size: 0.2
  random_state: 42

models:
  - name: random_forest
    tune: true
    params:
      n_estimators: [100, 300]
      max_depth: [null, 10, 30]

  - name: logistic_regression
    tune: false

tuning:
  method: grid
  cv: 5
  scoring: auto

output:
  experiments_dir: ./experiments
  save_model: true


## Fonctionnement du pipeline AutoML

Chargement des données (data_loader.py)

Lecture de .data, .solution, .type

Création automatique des noms de colonnes

Séparation numérique / catégorielle

Détection de la tâche (task_detector.py)

Analyse de la variable cible (y)

Détermination automatique : classification ou regression

Prétraitement (preprocess.py)

Imputation des valeurs manquantes

Encodage des colonnes catégorielles (OneHotEncoder)

Standardisation des colonnes numériques (StandardScaler)

Entraînement (runner.py)

Split train/validation/test

Entraînement de plusieurs modèles (models.py)

Optimisation d’hyperparamètres (tuner.py)

Évaluation (metrics.py)

Sélection automatique des métriques (accuracy, F1, R², etc.)

Sauvegarde des résultats et meilleurs modèles

Persistance et reproductibilité

Modèles enregistrés dans experiments/run_*/

Logs et configuration stockés pour chaque run

Reproductibilité assurée via seed et dépendances fixes

## Exemple de sortie (experiments/run_2025_10_09_1530/)

experiments/run_2025_10_09_1530/
├── config.yaml
├── model.joblib
├── preprocessor.joblib
├── metrics.csv
├── results.json
└── log.txt
Contenu metrics.csv :

metric	value
accuracy	0.842
f1_macro	0.817
roc_auc	0.885

# Tests unitaires
Les tests se trouvent dans le dossier tests/.
Exemple pour tester le module de chargement :


pytest tests/test_data_loader.py

# Livrables
Livrable	Description
- Package Python	Code complet dans src/automl, installable via pip
- Article scientifique	Présente la méthodologie, les expériences et les résultats
- Présentation orale (10 min)	Résumé visuel du projet et des résultats
- Dossier d’expériences	Contient les modèles, métriques et logs

# Bonnes pratiques
Fixer les seeds (numpy, random, sklearn)
Rendre chaque run reproductible (sauvegarder config.yaml)
Ne jamais modifier les données fournies
Respecter les limites de ressources du cluster Skinner
Écrire un code clair, commenté et modulaire

# Perspectives
Ce code servira de base pour le module DevOps du semestre suivant.
Les améliorations prévues :
Intégration d’Optuna pour le tuning plus intelligent
Support de modèles avancés (XGBoost, LightGBM)
Intégration d’un suivi d’expériences avec MLflow
Génération automatique de rapports (Markdown ou PDF)

# Auteurs
Projet réalisé par :
...

Sous la supervision de :
...