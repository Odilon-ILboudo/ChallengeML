import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



class Preprocess:
    def __init__(self, path):
    
        self.path = path
        # Attributs à remplir après load()
        self.data = None
        self.labels = None
        self.types = None
        self.name = None
        self.numerical_index = None
        self.categorical_index = None
        self.binary_index = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.validation_data = None
        self.validation_labels = None
    
    
    @staticmethod
    def load_dataset(path):
        if not os.path.isdir(path):
            print("Erreur: le chemin donné n'est pas un dossier ou c'est un chemin qui n'existe pas.")
            return None
        
        folder = os.path.basename(os.path.normpath(path))
        if not folder.startswith("data_"):
            print("Erreur: veuillez donner un dossier de cette forme (data_<Lettre>).")
            return None
        
        if any(os.path.isdir(os.path.join(path, x)) for x in os.listdir(path)):
            print("Veuillez insérer le chemin d'un dossier qui contient directement les fichiers .data, .solution et .type")
            return None
        
        files = os.listdir(path)
        data_files = [f for f in files if f.endswith(".data")]
        sol_files  = [f for f in files if f.endswith(".solution")]
        type_files = [f for f in files if f.endswith(".type")]
        
        if len(data_files) != 1 or len(sol_files) != 1 or len(type_files) != 1:
            print("Erreur: le dossier doit contenir exactement 1 fichier .data, 1 fichier .solution et 1 fichier .type.")
            return None
        
        name = data_files[0].replace(".data", "")
        data_path = os.path.join(path, name + ".data")
        solution_path = os.path.join(path, name + ".solution")
        type_path = os.path.join(path, name + ".type")
        
        
        # Détection simple du genre de data avec ":" comme (data_F) pour éviter ParserError
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            if ":" in first_line:
                print("Erreur: dataset sparse détecté (format index:val). Ce loader (pandas) ne gère pas ce genre de Dataset pour l'instant.")
                return None
        
        
        data = pd.read_csv(data_path, sep=r"\s+", header=None)
        labels = pd.read_csv(solution_path, sep=r"\s+", header=None)
        types = pd.read_csv(type_path, header=None)[0].tolist()
        
        return data, labels, types, name
    
    # Charger Notre DataSet en faisons appel à la fonction static load_dataset avec un path valid
    # Dans le cas de path invalid des message d'erreur s'affiche 
    # Aprés le chaqrgement du DataSet le csv des fichier .data et .solution et .type est affecter respectivement aux attribut (data, labels, types)
    def load(self):
        result = Preprocess.load_dataset(self.path)
        if result is None:
            return None
        
        self.data, self.labels, self.types, self.name = result
        return result
    
    
    # Detection du type de la problématique
    def detect_task_type(self):
        if self.labels is None:
            print("Erreur: appelle load() avant detect_task_type().")
            return None
        
        y = self.labels.values
        n_cols = y.shape[1]
        
        # 1 colonne 
        if n_cols == 1:
            y_1d = self.labels.iloc[:, 0].to_numpy()
            target_type = type_of_target(y_1d)
            
            if target_type == "binary":
                return "binary"
            
            if target_type == "continuous":
                return "regression"
            
            if target_type == "multiclass":
                # Ici on gére le cas ou on Corriger le cas de "régression stockée en int"
                uniq = np.unique(y_1d)
                n = len(y_1d)
                if len(uniq) > max(20, int(0.05 * n)):
                    return "regression"
                return "multiclass"
            
            return f"inconnu_{target_type}"
        
        # plusieurs colonnes 
        target_type = type_of_target(y)
        
        if target_type == "multilabel-indicator":
            row_sums = y.sum(axis=1)
            if np.all(row_sums == 1):
                return "multiclass_onehot"
            return "multilabel"
        
        if target_type == "continuous-multioutput":
            return "regression_multioutput"
        
        return f"inconnu_{target_type}"
    
    
    def get_dataset_info(self):
        if self.data is None or self.labels is None or self.types is None:
            print("Erreur: appelle load() avant get_dataset_info().")
            return None
        
        # indices (au cas où pas encore calculés)
        if self.numerical_index is None or self.categorical_index is None or self.binary_index is None:
            self.numerical_index, self.categorical_index, self.binary_index = self.get_feature_type_indices()
        
        task = self.detect_task_type()
        
        info = {
            "nom_du_dataset": self.name,
            
            ".data_shape": self.data.shape,          # shape du .data
            ".solution_shape": self.labels.shape,    # shape du .solution
            ".type_shape": len(self.types),      # taille du .type (nb de features décrites)
            
            # Détail des types de features
            "nombre_numerical_feature": len(self.numerical_index),
            "nombre_categorical_feature": len(self.categorical_index),
            "nombre_binary_feature": len(self.binary_index),
            
            # Missing values les valeurs Nan
            "missing_in_data": int(self.data.isna().sum().sum()),
            "missing_in_solution": int(self.labels.isna().sum().sum()),
            
            # Type de tâche
            "task_type": task,
        }
        
        # Distribution des classes / labels
        if task in ["binary", "multiclass"]:
            info["class_distribution"] = self.labels.iloc[:, 0].value_counts().to_dict()
        
        elif task == "multiclass_onehot":
            y_class = self.labels.values.argmax(axis=1)
            unique, counts = np.unique(y_class, return_counts=True)
            info["class_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}
        
        elif task == "multilabel":
            d = {}
            for j in range(self.labels.shape[1]):
                d[f"label_{j}"] = int(self.labels.iloc[:, j].sum())
            info["label_distribution"] = d
            info["class_distribution"] = None
        
        elif task in ["regression", "regression_multioutput"]:
            info["class_distribution"] = None
        
        else:
            info["class_distribution"] = None
        
        return info
    
    
    # On itére sur la liste types de features pour retourner la liste des index pour chaque type (numerical ou categorical ou binary)
    def get_feature_type_indices(self):
        numerical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "numerical"]
        categorical_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "categorical"]
        binary_index = [i for i, t in enumerate(self.types) if str(t).strip().lower() == "binary"]
        return numerical_index, categorical_index, binary_index
    
    # Stratégie de nettoyage + encodage (+ normalisation optionnelle)
    def build_preprocessor(self, scale_numeric=True):
        self.numerical_index, self.categorical_index, self.binary_index = self.get_feature_type_indices()
        
        # Si le type de features et numeric on applique la mediane sur les donner Nan
        # StandardScaler() pour centrer et reduire (Optionnel)
        if scale_numeric:
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            # Optionnel car certains model en ont besoin et D'autres non (RandomForest, GradientBoosting)
            # 
        else:
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ])
        
        # Encodage des features categorical en utilisant la methode de OneHotEncoder 
        # Remplacement des Nan par des valeur en utilisant la methode most_frequent
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        
        # Pas besoin d'encodage car c'est deja des chiffre 0 et 1 
        binary_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        
        transformers = []
        if len(self.numerical_index) > 0:
            transformers.append(("num", numerical_pipeline, self.numerical_index))
        if len(self.categorical_index) > 0:
            transformers.append(("cat", categorical_pipeline, self.categorical_index))
        if len(self.binary_index) > 0:
            transformers.append(("bin", binary_pipeline, self.binary_index))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor
    
    
    # Split (train/valid/test) après load, avant fit du preprocessing pour eviter le data leakage
    def split(self, test_size=0.2, validation_size=0.2, random_state=42):
        
        if self.data is None or self.labels is None:
            print("Erreur: appelle load() avant split().")
            return None
        
        temp_size = test_size + validation_size
        self.train_data, X_temp, self.train_labels, y_temp = train_test_split(self.data, self.labels, test_size=temp_size, random_state=random_state)
        
        validation_ratio = validation_size / temp_size
        self.validation_data, self.test_data, self.validation_labels, self.test_labels = train_test_split(X_temp, y_temp, test_size=(1 - validation_ratio), random_state=random_state)
        
        return self.train_data, self.validation_data, self.test_data, self.train_labels, self.validation_labels, self.test_labels
