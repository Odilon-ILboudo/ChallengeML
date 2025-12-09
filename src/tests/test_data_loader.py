import sys
import os
import pandas as pd

# Ajouter le dossier src au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from automl.data_loader import load_dataset
from automl_project.src.automl.main_preprocess import clean_data

base_path = "C:/Users/l15/Documents/M1_IA/Methodologie_IA/projet/ChallengeMachineLearning/data/data_G/data_G"

X, y, types = load_dataset(base_path)
X_clean, preprocessor = clean_data(X, types)


print("Données originales :", X.head(10))
print("Target originales :", y.head(10))
print("Longueur des Données originales :", X.shape)
print("Données transformées :", X_clean)
print("Longueur des Données transformées :", X_clean.shape)

#X_clean_a = pd.DataFrame(X_clean[:10, :20].toarray())
#print(X_clean_a)
