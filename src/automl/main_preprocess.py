import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Import de data_loader et detectproblem
from data_loader import load_data_files
from detectproblem import detect_problem_type


# Fonction de préprocessing
def clean_data(X: pd.DataFrame, feature_types: list):
    """
    Nettoie et encode les données selon leur type.
    Retourne la matrice transformée (numpy ou sparse matrix) et le préprocesseur.
    """

    num_cols = [c for c, t in zip(X.columns, feature_types) if "num" in t.lower() or t == "Numerical"]
    cat_cols = [c for c, t in zip(X.columns, feature_types) if "cat" in t.lower() or t == "Categorical"]

    # Pipeline numérique
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline catégoriel
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combinaison
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_clean = preprocessor.fit_transform(X)

    return X_clean, preprocessor



# SCRIPT PRINCIPAL
if __name__ == "__main__":

    # Dossier contenant les fichiers .data/.solution/.type
    folder_path = "/info/corpus/ChallengeMachineLearning/data_B/"

    # Chargement des données
    X, y, types = load_data_files(folder_path)

    if X is None or y is None or types is None:
        raise ValueError("Impossible de charger les fichiers data/solution/type")

    print("Données chargées :")
    print(" - X :", X.shape)
    print(" - y :", y.shape)
    print(" - Types :", types.shape)

    # Détection du type de problème
    problem_type, is_multilabel = detect_problem_type(y)
    print(f"\n Type de problème détecté : {problem_type}, multilabel = {is_multilabel}")

    # Extraction de la liste brute des types
    feature_types = types.iloc[:, 0].tolist()

    # Nettoyage + encodage
    X_clean, preprocessor = clean_data(X, feature_types)
    print("\n Données prétraitées :", X_clean.shape)

    # Conversion sparse → dense
    if hasattr(X_clean, "toarray"):
        X_clean = X_clean.toarray()

    # Conversion en DataFrame
    X_clean_df = pd.DataFrame(X_clean)

    # Ajout des colonnes target
    for col in y.columns:
        X_clean_df[col] = y[col]

    # Sauvegarde
    save_dir = os.path.join(os.getcwd(), "dataOut")
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "dataset_preprocessed.csv")

    X_clean_df.to_csv(output_path, index=False)

    print(f"\n Dataset prétraité sauvegardé dans : {output_path}")
