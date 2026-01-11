class ModelesConditions:
    def __init__(self, random_state=42):
        self.random_state = int(random_state)

    def _ajouter_modele(self, modeles, nom, estimator, task_types, accepte_sparse=True, besoin_normalisation=False, pca_ok=True, svd_ok=True, poly_ok=False, has_predict_proba=False, priorite=50, params_light=None, params_full=None):
        modeles[str(nom)] = {"estimator": estimator, "task_types": tuple(task_types), "accepte_sparse": bool(accepte_sparse), "besoin_normalisation": bool(besoin_normalisation), "pca_ok": bool(pca_ok), "svd_ok": bool(svd_ok), "poly_ok": bool(poly_ok), "has_predict_proba": bool(has_predict_proba), "priorite": int(priorite), "params_light": dict(params_light) if isinstance(params_light, dict) else {}, "params_full": dict(params_full) if isinstance(params_full, dict) else {}}

    def get_modeles(self, task_type):
        rs = self.random_state
        t = str(task_type).strip().lower() if task_type is not None else None
        modeles = {}

        # Classification
        if t == "multiclass_onehot":
            from sklearn.linear_model import SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC
            from sklearn.linear_model import PassiveAggressiveClassifier

            self._ajouter_modele(
                modeles,
                "SGDClassifier",
                SGDClassifier(random_state=rs, max_iter=1200, tol=1e-2, early_stopping=True, n_iter_no_change=5),
                task_types=("multiclass_onehot",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=10,
                params_light={
                    "loss": ["log_loss"],
                    "alpha": [1e-4, 1e-3],
                    "penalty": ["l2"],
                    "class_weight": [None, "balanced"]
                },
                params_full={
                    "loss": ["log_loss", "hinge"],
                    "alpha": [1e-5, 1e-4, 1e-3],
                    "penalty": ["l2", "elasticnet"],
                    "l1_ratio": [0.15, 0.5],
                    "class_weight": [None, "balanced"]
                }
            )


            self._ajouter_modele(
                modeles,
                "RidgeClassifier",
                RidgeClassifier(random_state=rs),
                task_types=("multiclass_onehot",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=15,
                params_light={"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles,
                "LinearSVC",
                LinearSVC(random_state=rs, max_iter=2000, tol=1e-2, dual=False),
                task_types=("multiclass_onehot",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=20,
                params_light={},  # pas de BayesSearch en light pour que ça prend pas trop de temps 
                params_full={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}
            )


            self._ajouter_modele(
                modeles,
                "PassiveAggressive",
                PassiveAggressiveClassifier(random_state=rs, max_iter=2000, tol=1e-3),
                task_types=("multiclass_onehot",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=25,
                params_light={"C": [0.1, 1.0, 10.0], "loss": ["hinge", "squared_hinge"], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "loss": ["hinge", "squared_hinge"], "class_weight": [None, "balanced"]}
            )

            return modeles

        if t in ("binary", "multiclass"):
            from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC

            self._ajouter_modele(
                modeles,
                "LogisticRegression",
                LogisticRegression(max_iter=2000, random_state=rs, solver="saga"),
                task_types=("binary", "multiclass"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=True,
                priorite=10,
                params_light={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l2", "l1", "elasticnet"], "l1_ratio": [0.15, 0.5, 0.85], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles,
                "LinearSVC",
                LinearSVC(random_state=rs, max_iter=5000),
                task_types=("binary", "multiclass"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=15,
                params_light={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles,
                "SGDClassifier",
                SGDClassifier(random_state=rs, max_iter=2000, tol=1e-3),
                task_types=("binary", "multiclass"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=20,
                params_light={"loss": ["hinge", "log_loss"], "alpha": [1e-5, 1e-4, 1e-3], "penalty": ["l2", "elasticnet"], "class_weight": [None, "balanced"]},
                params_full={"loss": ["hinge", "log_loss"], "alpha": [1e-6, 1e-5, 1e-4, 1e-3], "penalty": ["l2", "l1", "elasticnet"], "l1_ratio": [0.15, 0.5, 0.85], "class_weight": [None, "balanced"]}
            )

            self._ajouter_modele(
                modeles,
                "RidgeClassifier",
                RidgeClassifier(random_state=rs),
                task_types=("binary", "multiclass"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=25,
                params_light={"alpha": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
                params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}
            )

            # Modèles non linéaires (souvent mieux sans normalisation, et évite sparse)
            try:
                from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier

                self._ajouter_modele(
                    modeles,
                    "ExtraTrees",
                    ExtraTreesClassifier(random_state=rs, n_jobs=-1),
                    task_types=("binary", "multiclass"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=True,
                    priorite=55,
                    params_light={"n_estimators": [500], "max_depth": [None, 20], "min_samples_leaf": [1, 2], "max_features": ["sqrt"], "class_weight": [None, "balanced"]},
                    params_full={"n_estimators": [500, 900], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None], "class_weight": [None, "balanced"]}
                )

                self._ajouter_modele(
                    modeles,
                    "RandomForest",
                    RandomForestClassifier(random_state=rs, n_jobs=-1),
                    task_types=("binary", "multiclass"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=True,
                    priorite=60,
                    params_light={"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 4], "max_features": ["sqrt"], "class_weight": [None, "balanced"]},
                    params_full={"n_estimators": [300, 600, 900], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None], "bootstrap": [True, False], "class_weight": [None, "balanced"]}
                )

                self._ajouter_modele(
                    modeles,
                    "HistGradientBoosting",
                    HistGradientBoostingClassifier(random_state=rs),
                    task_types=("binary", "multiclass"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=True,
                    priorite=65,
                    params_light={"learning_rate": [0.05, 0.1], "max_iter": [200, 500], "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 50]},
                    params_full={"learning_rate": [0.01, 0.05, 0.1], "max_iter": [200, 500, 1000], "max_leaf_nodes": [31, 63, 127], "min_samples_leaf": [10, 20, 50, 100], "l2_regularization": [0.0, 0.1, 1.0, 10.0]}
                )

            except Exception:
                pass

            return modeles

        # Régression
        if t in ("regression", "regression_multioutput"):
            from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
            from sklearn.svm import LinearSVR
            from sklearn.multioutput import MultiOutputRegressor

            ridge_est = Ridge()
            sgd_est = SGDRegressor(random_state=rs, max_iter=2000, tol=1e-3)
            enet_est = ElasticNet(random_state=rs, max_iter=2000)
            lsvr_est = LinearSVR()

            if t == "regression_multioutput":
                sgd_est = MultiOutputRegressor(sgd_est)
                enet_est = MultiOutputRegressor(enet_est)
                lsvr_est = MultiOutputRegressor(lsvr_est)

            p = "estimator__" if t == "regression_multioutput" else ""

            self._ajouter_modele(
                modeles,
                "Ridge",
                ridge_est,
                task_types=("regression", "regression_multioutput"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=10,
                params_light={"alpha": [0.1, 1.0, 10.0]},
                params_full={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            )

            self._ajouter_modele(
                modeles,
                "SGDRegressor",
                sgd_est,
                task_types=("regression", "regression_multioutput"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=20,
                params_light={p + "alpha": [1e-5, 1e-4, 1e-3], p + "loss": ["squared_error", "huber"], p + "penalty": ["l2", "elasticnet"]},
                params_full={p + "alpha": [1e-6, 1e-5, 1e-4, 1e-3], p + "loss": ["squared_error", "huber"], p + "penalty": ["l2", "l1", "elasticnet"], p + "l1_ratio": [0.15, 0.5, 0.85]}
            )

            self._ajouter_modele(
                modeles,
                "LinearSVR",
                lsvr_est,
                task_types=("regression", "regression_multioutput"),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=True,
                poly_ok=True,
                has_predict_proba=False,
                priorite=25,
                params_light={p + "C": [0.1, 1.0, 10.0], p + "epsilon": [0.0, 0.1, 0.2]},
                params_full={p + "C": [0.01, 0.1, 1.0, 10.0, 100.0], p + "epsilon": [0.0, 0.05, 0.1, 0.2, 0.4]}
            )

            self._ajouter_modele(
                modeles,
                "ElasticNet",
                enet_est,
                task_types=("regression", "regression_multioutput"),
                accepte_sparse=False,
                besoin_normalisation=True,
                pca_ok=True,
                svd_ok=False,
                poly_ok=True,
                has_predict_proba=False,
                priorite=30,
                params_light={p + "alpha": [0.1, 1.0, 10.0], p + "l1_ratio": [0.1, 0.5, 0.9]},
                params_full={p + "alpha": [0.01, 0.1, 1.0, 10.0, 100.0], p + "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
            )

            # Modèles non linéaires (évite sparse)
            try:
                from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor

                hgb_est = HistGradientBoostingRegressor(random_state=rs)
                if t == "regression_multioutput":
                    hgb_est = MultiOutputRegressor(hgb_est)

                self._ajouter_modele(
                    modeles,
                    "ExtraTreesRegressor",
                    ExtraTreesRegressor(random_state=rs, n_jobs=-1),
                    task_types=("regression", "regression_multioutput"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=False,
                    priorite=55,
                    params_light={"n_estimators": [500], "max_depth": [None, 20], "min_samples_leaf": [1, 2], "max_features": ["sqrt", None]},
                    params_full={"n_estimators": [500, 900], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None]}
                )

                self._ajouter_modele(
                    modeles,
                    "RandomForestRegressor",
                    RandomForestRegressor(random_state=rs, n_jobs=-1),
                    task_types=("regression", "regression_multioutput"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=False,
                    priorite=60,
                    params_light={"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 4], "max_features": ["sqrt"]},
                    params_full={"n_estimators": [300, 600, 900], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None], "bootstrap": [True, False]}
                )

                self._ajouter_modele(
                    modeles,
                    "HistGradientBoostingRegressor",
                    hgb_est,
                    task_types=("regression", "regression_multioutput"),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=False,
                    priorite=65,
                    params_light={"learning_rate": [0.05, 0.1], "max_iter": [200, 500], "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 50]},
                    params_full={"learning_rate": [0.01, 0.05, 0.1], "max_iter": [200, 500, 1000], "max_leaf_nodes": [31, 63, 127], "min_samples_leaf": [10, 20, 50, 100], "l2_regularization": [0.0, 0.1, 1.0, 10.0]}
                )

            except Exception:
                pass

            return modeles

        # Multilabel
        if t == "multilabel":
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
            from sklearn.svm import LinearSVC

            self._ajouter_modele(
                modeles,
                "OVR_LogisticRegression",
                OneVsRestClassifier(LogisticRegression(max_iter=800, tol=1e-3, random_state=rs, solver="saga", penalty="l2"), n_jobs=-1),
                task_types=("multilabel",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=False,
                svd_ok=True,
                poly_ok=False,
                has_predict_proba=True,
                priorite=30,
                params_light={"estimator__C": [0.5, 1.0]},
                params_full={"estimator__C": [0.1, 1.0, 10.0]}
            )

            self._ajouter_modele(
                modeles,
                "OVR_LinearSVC",
                OneVsRestClassifier(LinearSVC(random_state=rs, max_iter=2000, tol=1e-3, dual="auto"), n_jobs=-1),
                task_types=("multilabel",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=False,
                svd_ok=True,
                poly_ok=False,
                has_predict_proba=False,
                priorite=25,
                params_light={"estimator__C": [1.0]},
                params_full={"estimator__C": [0.1, 1.0, 10.0]}
            )

            self._ajouter_modele(
                modeles,
                "OVR_SGDClassifier",
                OneVsRestClassifier(SGDClassifier(random_state=rs, max_iter=1200, tol=1e-2,early_stopping=True,n_iter_no_change=5), n_jobs=-1),
                
                task_types=("multilabel",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=False,
                svd_ok=True,
                poly_ok=False,
                has_predict_proba=False,
                priorite=10,
                params_light={"estimator__loss": ["log_loss"], "estimator__alpha": [1e-4, 1e-3], "estimator__penalty": ["l2"]},
                params_full={"estimator__loss": ["log_loss", "hinge"], "estimator__alpha": [1e-5, 1e-4, 1e-3], "estimator__penalty":["l2", "elasticnet"], "estimator__l1_ratio": [0.15, 0.5]}
                )


            self._ajouter_modele(
                modeles,
                "OVR_RidgeClassifier",
                OneVsRestClassifier(RidgeClassifier(random_state=rs), n_jobs=-1),
                task_types=("multilabel",),
                accepte_sparse=True,
                besoin_normalisation=True,
                pca_ok=False,
                svd_ok=True,
                poly_ok=False,
                has_predict_proba=False,
                priorite=25,
                params_light={"estimator__alpha": [0.1, 1.0, 10.0]},
                params_full={"estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            )

            # Modèles natifs multilabel (évite sparse)
            try:
                from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

                self._ajouter_modele(
                    modeles,
                    "RF_Multilabel_Native",
                    RandomForestClassifier(random_state=rs),
                    task_types=("multilabel",),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=True,
                    priorite=60,
                    params_light={"n_estimators": [400], "max_depth": [None, 20], "min_samples_leaf": [1, 4], "max_features": ["sqrt"]},
                    params_full={"n_estimators": [400, 800], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None]}
                )

                self._ajouter_modele(
                    modeles,
                    "ET_Multilabel_Native",
                    ExtraTreesClassifier(random_state=rs, n_jobs=-1),
                    task_types=("multilabel",),
                    accepte_sparse=False,
                    besoin_normalisation=False,
                    pca_ok=False,
                    svd_ok=False,
                    poly_ok=False,
                    has_predict_proba=True,
                    priorite=55,
                    params_light={"n_estimators": [600], "max_depth": [None, 20], "min_samples_leaf": [1, 2], "max_features": ["sqrt"]},
                    params_full={"n_estimators": [600, 1000], "max_depth": [None, 10, 20, 40], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None]}
                )

            except Exception:
                pass

            return modeles

        return modeles

    def trier_par_priorite(self, modeles):
        if not isinstance(modeles, dict):
            return []
        return sorted(modeles.items(), key=lambda kv: int(kv[1].get("priorite", 50)))
