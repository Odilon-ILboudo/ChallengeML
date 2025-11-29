import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    r2_score, jaccard_score
)
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


class ChallengeML:
    def __init__(self):
        self.data = None
        self.solution = None
        self.type = None
        self.model_info = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    # ===== PUBLIQUE =====
    def fit(self, path_repertory: str):
        self._load_file(path_repertory)


        self._normalize_data()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.solution, test_size=0.2, random_state=42)

        self.model_info = self._detect_models_and_metrics()
        print(self.model_info)

    def eval(self):

        print("Tâche détectée :", self.model_info["type"])
        print("Modèles candidats :", [type(m).__name__ for m in self.model_info["models"]])
        print("Métriques :", self.model_info["metrics"])

        result = self._train_and_select_best_model()
        print(
            f"Meilleur modèle : {type(result['best_model']).__name__} ({result['metric_used']} = {result['best_score']:.4f})")

    # ===== PRIVÉE =====
    #  Chargement des fichiers et lectures des fichiers
    def _load_file(self, path_repertory: str):
        #  Lecture des chemins
        letter_of_repertory = path_repertory[-1]
        path_data = os.path.join(path_repertory, "data_" + letter_of_repertory + ".data")
        path_type = os.path.join(path_repertory, "data_" + letter_of_repertory + ".type")
        path_solution = os.path.join(path_repertory, "data_" + letter_of_repertory + ".solution")

        #  Lecture des données
        self.data = pd.read_csv(path_data, sep=r'\s+', header=None)
        self.solution = pd.read_csv(path_solution, sep=r'\s+', header=None)
        self.type = pd.read_csv(path_type, sep=r'\s+', header=None, names=['Type'])


    #  Normalisation des données
    def _normalize_data(self):
        # Forcer noms de colonnes en str
        self.data.columns = self.data.columns.astype(str)

        # Mapping col -> type
        type_map = dict(zip(self.data.columns, self.type["Type"]))

        # Séparer colonnes selon type
        numeric_features = [col for col, t in type_map.items() if t.lower() == "numerical"]
        categorical_features = [col for col, t in type_map.items() if t.lower() == "categorical"]
        bool_features = [col for col, t in type_map.items() if t.lower() == "boolean"]

        numeric_transformer = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        bool_transformer = Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
        ])


        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
                ("bool", bool_transformer, bool_features)
            ],
            remainder="drop"
        )

        # Pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])


        # Transformation
        X_transformed = pipeline.fit_transform(self.data)


        # Récupérer noms des colonnes transformées
        cat_cols = pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(
            categorical_features) if categorical_features else []
        bool_cols = pipeline.named_steps["preprocessor"].transformers_[2][1].get_feature_names_out(
            bool_features) if bool_features else []
        new_columns = numeric_features + list(cat_cols) + list(bool_cols)


        self.X = pd.DataFrame(X_transformed, columns=new_columns)


    def _detect_models_and_metrics(self):
        """
        Détecte le type de tâche à partir de y et renvoie :
          - le type de tâche
          - la liste de modèles sklearn à tester
          - la liste de métriques adaptées
        """

        n_cols = self.solution.shape[1]

        # ----- Cas 1 : plusieurs colonnes -----
        if n_cols > 1:
            unique_vals = np.unique(self.solution.values)
            if np.all(np.isin(unique_vals, [0, 1])):
                # Vérifie si chaque ligne a un seul 1
                one_hot = np.all(self.solution.sum(axis=1).isin([0, 1]))
                if one_hot:
                    task = "classification_multi-classe"
                    models = [
                        HistGradientBoostingClassifier(),
                        MLPClassifier(max_iter=500),
                        LogisticRegression(max_iter=1000, multi_class="multinomial")
                    ]
                    metrics = ["accuracy", "f1_macro"]
                else:
                    task = "classification_multi-label"
                    models = [
                        MultiOutputClassifier(HistGradientBoostingClassifier()),
                        MultiOutputClassifier(RandomForestClassifier()),
                        MultiOutputClassifier(MLPClassifier(max_iter=500))
                    ]
                    metrics = ["f1_micro", "f1_macro", "jaccard"]
            else:
                task = "régression_multi-sortie"
                models = [
                    HistGradientBoostingRegressor(),
                    RandomForestRegressor()
                ]
                metrics = ["rmse", "r2", "mae"]

        # ----- Cas 2 : une seule colonne -----
        else:
            col = self.solution.iloc[:, 0]
            if np.issubdtype(col.dtype, np.integer):
                unique_vals = np.unique(col)
                if len(unique_vals) == 2:
                    task = "classification_binaire"
                    models = [
                        HistGradientBoostingClassifier(),
                        LogisticRegression(max_iter=1000),
                        MLPClassifier(max_iter=500)
                    ]
                    metrics = ["f1", "roc_auc"]
                else:
                    task = "classification_multi-classe"
                    models = [
                        HistGradientBoostingClassifier(),
                        MLPClassifier(max_iter=500),
                        LogisticRegression(max_iter=1000, multi_class="multinomial")
                    ]
                    metrics = ["accuracy", "f1_macro"]
            else:
                task = "régression"
                models = [
                    HistGradientBoostingRegressor(),
                    RandomForestRegressor()
                ]
                metrics = ["rmse", "r2", "mae"]

        return {"type": task, "models": models, "metrics": metrics}

    def _train_and_select_best_model(self):
        """
        Essaie plusieurs modèles et sélectionne celui avec le meilleur score
        selon les métriques recommandées.
        """

        best_model = None
        best_score = -np.inf
        best_metric = None

        for model in self.model_info["models"]:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            for metric in self.model_info["metrics"]:
                if metric in ["accuracy"]:
                    score = accuracy_score(self.y_test, y_pred)
                elif metric in ["f1", "f1_macro"]:
                    score = f1_score(self.y_test, y_pred, average="macro")
                elif metric == "f1_micro":
                    score = f1_score(self.y_test, y_pred, average="micro")
                elif metric == "roc_auc":
                    # Pour éviter les erreurs si prédictions non binaires
                    try:
                        score = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                    except:
                        score = f1_score(self.y_test, y_pred, average="macro")
                elif metric == "jaccard":
                    score = jaccard_score(self.y_test, y_pred, average="samples")
                elif metric == "rmse":
                    score = -np.sqrt(mean_squared_error(self.y_test, y_pred))
                elif metric == "r2":
                    score = r2_score(self.y_test, y_pred)
                elif metric == "mae":
                    score = -np.mean(np.abs(self.y_test - y_pred))
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metric = metric

        return {"best_model": best_model, "best_score": best_score, "metric_used": best_metric}






