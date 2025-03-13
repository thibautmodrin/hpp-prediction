import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.data.pandas_dataset
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import (
    recall_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    precision_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import make_scorer

# # Création du dossier de cache si nécessaire
# if not os.path.exists("cache_directory"):
#     os.makedirs("cache_directory")

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)


    # Fonction pour mettre à jour les listes de features en fonction du dictionnaire de variables
    def update_list_features(X_train, df_dico):
        df_dico = df_dico.reindex(X_train.columns)
        quant_features = df_dico[df_dico['type'] == 'quantitative'].index.tolist()
        binary_features = df_dico[df_dico['type'] == 'binaire'].index.tolist()
        nominal_features = df_dico[df_dico['type'] == 'nominale'].index.tolist()
        ordinal_features = df_dico[df_dico['type'] == 'ordinale'].index.tolist()
        return quant_features, binary_features, nominal_features, ordinal_features

    def custom_recall_precision_scorer(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        score = 0.8 * recall + 0.2 * precision
        return score

    custom_scorer = make_scorer(custom_recall_precision_scorer)


    # Séparation des features et de la target
    df_dico = pd.read_csv('s3://openlab-mlfow-bucket/final_project/desc_bis.csv',index_col=0)
    df_model = pd.read_csv('s3://openlab-mlfow-bucket/final_project/df_avt_acc_bis.csv', index_col=0)

    X=df_model.drop(['hpp_trans','hta_gest','hta_chro','poids_mere'],axis=1)
    # X=df_model[variable_test]
    y = df_model["hpp_trans"]

    # Séparation en Train (80%) et Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train = X_train.astype('float64')
    y_train = y_train.astype('float64')
    X_test = X_test.astype('float64')
    y_test = y_test.astype('float64')

    quant_features, binary_features, nominal_features, ordinal_features = (
        update_list_features(X_train, df_dico)
    )
    logger.info(f"Train : {X_train.shape}, Test : {X_test.shape}")

    # Définition des transformations avec imputation

    # Pour les variables quantitatives : imputation par la moyenne puis standardisation
    quantitative_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Pour les variables ordinales : imputation par la modalité la plus fréquente puis encodage ordinal
    ordinal_transformer = Pipeline(steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])

    # Pour les variables nominales : imputation par la modalité la plus fréquente puis encodage OneHot
    nominal_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )

    # Préprocessing via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("quant", quantitative_transformer, quant_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("nom", nominal_transformer, nominal_features),
            ("bin", "passthrough", binary_features),
        ]
    )

    # Pipeline avec SMOTE et LogisticRegression
    smote = SMOTE(random_state=42)
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            # ("balancing", BorderlineSMOTE(random_state=42)),
            # ('oversampler', SMOTETomek(random_state=42)),
            ('oversampler', SMOTEENN(random_state=42)),
            # ('oversampler', SMOTE(random_state=42)),
            # ('oversampler', BorderlineSMOTE(random_state=42)),
            ("classifier", LogisticRegression(random_state=42,max_iter=1000)),
        ],
        # memory="cache_directory",
    )
    param_grid = [
        {
            'classifier__solver': ['lbfgs'],
            'classifier__penalty': ['l2', None],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'classifier__class_weight': ['balanced', None],
            'oversampler__sampling_strategy': [0.1, 0.25, 0.5, 0.75, 1.0],
            # 'oversampler__k_neighbors': [3, 5, 7, 10]
        },
        {
            'classifier__solver': ['liblinear'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'classifier__class_weight': ['balanced', None],
            'oversampler__sampling_strategy': [0.1, 0.25, 0.5, 0.75, 1.0],
            # 'oversampler__k_neighbors': [3, 5, 7, 10]
        },
        {
            'classifier__solver': ['saga'],
            'classifier__penalty': ['elasticnet', 'l1', 'l2', None],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'classifier__class_weight': ['balanced', None],
            'oversampler__sampling_strategy': [0.1, 0.25, 0.5, 0.75, 1.0],
            # 'oversampler__k_neighbors': [3, 5, 7, 10]
        }
    ]
    # Définition des hyperparamètres pour GridSearchCV
    # param_grid = {

    #     "classifier__solver": ['lbfgs','liblinear','newton-cg','sag','saga'],
    #     "classifier__class_weight": ['balanced',None],
    #     "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    #     # "oversampler__sampling_strategy": [0.1,0.25,0.5,0.75,1],
    #     # "oversampler__k_neighbors": [2,5,10,20,40],



    #     # "classifier__penalty": ['l2'],
        
    #     # "classifier__class_weight": [{1:10},{1:20},{1:30},{1:40},{1:50},{1:60},{1:70},{1:80},{1:90},{1:100}],
    #     # "balancing__kind": ["borderline-1", "borderline-2"],
    #     # "balancing__sampling_strategy": [{1:10000},{1:20000},{1:30000},{1:40000},{1:50000},{1:60000},{1:70000},{1:80000},{1:90000},{1:100000}],
    #     # "balancing__k_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     # "classifier__class_weight": ['balanced',None],
    #     # "oversampler__sampling_strategy": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    #     # "oversampler__k_neighbors": [1,5,10,15,20,25,30,35,40,45,50],
        

    # }
    # Define custom scoring function
    # def custom_recall_precision_scorer(y_true, y_pred):
    #     recall = recall_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred)
    #     # Weighted average: 70% recall + 30% precision (adjust weights as needed)
    #     score = 0.7 * recall + 0.3 * precision
    #     return score

    # Create a scorer object
    # custom_scorer = make_scorer(custom_recall_precision_scorer)
    # Configuration de StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pour optimiser le recall
    lr_grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring = {
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),

        },
        refit='f1',
        cv=skf,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://thibautmodrin-mlflow.hf.space"))
    mlflow.set_experiment("HPP_Prediction_LOG_REG_RECALL")
    experiment = mlflow.get_experiment_by_name("HPP_Prediction_LOG_REG_RECALL")

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id, run_name="LR_SMOTEENN_penalty_gridsearch_f1")  # Ajoute run_name ici
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_id=run.info.run_id) as run:
        # Création et enregistrement du dataset d'entraînement
        train_df = pd.concat([X_train, y_train], axis=1)
        train_dataset = mlflow.data.from_pandas(
            train_df, name="train_dataset", targets="hpp_trans"
        )
        mlflow.log_input(train_dataset, context="training")

        # Entraînement de GridSearchCV
        logger.info("Début : Entraînement de GridSearchCV avec LogisticRegression...")
        lr_grid.fit(X_train, y_train)
        logger.info("Fin : Entraînement terminé.")

        best_index = lr_grid.best_index_


        best_model = lr_grid.best_estimator_
        mlflow.log_params(lr_grid.best_params_)
        mlflow.sklearn.log_model(best_model, "best_model_lr", input_example=X_train.head())
        mlflow.log_metric("cv_mean_f1", lr_grid.cv_results_["mean_test_f1"][best_index])
        mlflow.log_metric("cv_mean_precision", lr_grid.cv_results_["mean_test_precision"][best_index])
        mlflow.log_metric("cv_mean_recall", lr_grid.cv_results_["mean_test_recall"][best_index])
        mlflow.log_metric("cv_std_f1", lr_grid.cv_results_["std_test_f1"][best_index])
        mlflow.log_metric("cv_std_precision", lr_grid.cv_results_["std_test_precision"][best_index])
        mlflow.log_metric("cv_std_recall", lr_grid.cv_results_["std_test_recall"][best_index])



        # Évaluation finale sur le Test Set en testant plusieurs seuils
        y_prob_test = best_model.predict_proba(X_test)[:, 1]
        y_prob_train = best_model.predict_proba(X_train)[:, 1]

        seuils = np.arange(0.1, 1.0, 0.1)
        recalls_test = []
        precisions_test = []
        f1_scores_test = []

        for seuil in seuils:
            y_pred_test = (y_prob_test > seuil).astype(int)
            y_pred_train = (y_prob_train > seuil).astype(int)

            recall_test = recall_score(y_test, y_pred_test, zero_division=0)
            precision_test = precision_score(y_test, y_pred_test, zero_division=0)
            f1_test = f1_score(y_test, y_pred_test, zero_division=0)

            recalls_test.append(recall_test)
            precisions_test.append(precision_test)
            f1_scores_test.append(f1_test)

            mlflow.log_metric(f"recall_test_seuil_{seuil:.1f}", recall_test)
            mlflow.log_metric(f"precision_test_seuil_{seuil:.1f}", precision_test)
            mlflow.log_metric(f"f1_test_seuil_{seuil:.1f}", f1_test)

            logger.info(
                f"Seuil {seuil:.1f} - Recall Test: {recall_test:.3f}, Precision Test: {precision_test:.3f}, F1 Test: {f1_test:.3f}"
            )

        # Graphique des métriques en fonction du seuil
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(seuils, recalls_test, label="Recall", marker="o")
        ax.plot(seuils, precisions_test, label="Precision", marker="o")
        ax.plot(seuils, f1_scores_test, label="F1-Score", marker="o")
        ax.set_xlabel("Seuil de décision")
        ax.set_ylabel("Métrique")
        ax.set_title("Évolution des métriques en fonction du seuil (Test Set)")
        ax.legend()
        ax.grid(True)
        mlflow.log_figure(fig, "9-threshold_search.png")
        plt.close()

        # Matrices de confusion pour Train et Test (seuil 0.5)
        y_pred_test = (y_prob_test > 0.5).astype(int)
        y_pred_train = (y_prob_train > 0.5).astype(int)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, ax=ax)
        plt.title("Confusion Matrix - Train Set")
        mlflow.log_figure(fig, "1-confusion_matrix_train.png")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, ax=ax)
        plt.title("Confusion Matrix - Test Set")
        mlflow.log_figure(fig, "2-confusion_matrix_test.png")

        # Courbe ROC pour le Train Set
        fpr, tpr, _ = roc_curve(y_train, y_prob_train)
        roc_auc_train = roc_auc_score(y_train, y_prob_train)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_train:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Train Set")
        ax.legend()
        mlflow.log_figure(fig, "3-roc_curve_train.png")

        # Courbe ROC pour le Test Set
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        roc_auc_test = roc_auc_score(y_test, y_prob_test)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_test:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Test Set")
        ax.legend()
        mlflow.log_figure(fig, "4-roc_curve_test.png")

        # Courbes Precision-Recall
        train_precision, train_recall, _ = precision_recall_curve(y_train, y_prob_train)
        pr_auc_train = auc(train_recall, train_precision)
        fig, ax = plt.subplots()
        ax.plot(
            train_recall,
            train_precision,
            lw=2,
            label=f"PR curve (AUC = {pr_auc_train:.2f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve - Train Set")
        ax.legend()
        ax.grid()
        mlflow.log_figure(fig, "5-precision_recall_curve_train.png")

        test_precision, test_recall, _ = precision_recall_curve(y_test, y_prob_test)
        pr_auc_test = auc(test_recall, test_precision)
        fig, ax = plt.subplots()
        ax.plot(
            test_recall, test_precision, lw=2, label=f"PR curve (AUC = {pr_auc_test:.2f})"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve - Test Set")
        ax.legend()
        ax.grid()
        mlflow.log_figure(fig, "6-precision_recall_curve_test.png")

        # Feature Importance pour LogisticRegression
        try:
            feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
        except AttributeError:
            feature_names = X_train.columns
        coefficients = best_model.named_steps["classifier"].coef_[0]
        feature_importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "Absolute_Coefficient": np.abs(coefficients),
            }
        ).sort_values(by="Absolute_Coefficient", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            feature_importance_df["Feature"], feature_importance_df["Absolute_Coefficient"]
        )
        ax.set_xlabel("Absolute Coefficient Value")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance - LogisticRegression")
        ax.invert_yaxis()
        plt.tight_layout()
        mlflow.log_figure(fig, "7-feature_importance.png")

        # Rapport de classification
        report_train = classification_report(y_train, y_pred_train, zero_division=0)
        report_test = classification_report(y_test, y_pred_test, zero_division=0)
        mlflow.log_text(report_train, "8-classification_report_train.txt")
        mlflow.log_text(report_test, "9-classification_report_test.txt")


        # Rapport de classification imbalancé
        report_imbalanced = classification_report_imbalanced(y_test, y_pred_test)
        mlflow.log_text(report_imbalanced, "91-classification_report_imbalanced.txt")

        logger.info("✅ Modèle optimisé et validé enregistré sur MLflow")
        logger.info("📌 Meilleurs hyperparamètres : %s", lr_grid.best_params_)

    # Sauvegarde du meilleur modèle
    # mlflow.log_artifact("best_model_lr_skfold_robust.pkl")
