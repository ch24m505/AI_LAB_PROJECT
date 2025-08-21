#!/usr/bin/env python
# coding: utf-8
import argparse
import mlflow
import mlflow.spark
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    DecisionTreeClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from mlflow.tracking import MlflowClient


def promote_best_model(model_name, metric_name, current_run_id):
    """
    Compares the current model with the production model and promotes it
    if it performs better.
    """
    client = MlflowClient()
    current_run = client.get_run(current_run_id)
    current_metric = current_run.data.metrics.get(metric_name)

    if current_metric is None:
        print(f"Metric '{metric_name}' not found for current run. Skipping promotion.")
        return

    production_metric = -1

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            production_version = versions[0]
            production_run = client.get_run(production_version.run_id)
            production_metric = production_run.data.metrics.get(metric_name, -1)
    except mlflow.exceptions.RestException:
        print(f"Model '{model_name}' not found. Registering new model.")

    print(
        f"Current model metric: {current_metric:.4f}, "
        f"Production model metric: {production_metric:.4f}"
    )

    if current_metric > production_metric:
        print("New model is better! Promoting to Production.")
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
    else:
        print("Current model is not better than the production model.")


def train_model(model_name: str):
    """
    Loads ML-ready data, trains a selected model with hyperparameter tuning,
    and logs everything with MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name=f"{model_name}_tuning_run") as run:
        spark = SparkSession.builder.appName("TitanicModelTraining").getOrCreate()
        ml_ready_df = spark.read.parquet(
            "C:/Users/user/AI_LAB_PROJECT/data/processed/titanic_ml_ready"
        )
        train_data, test_data = ml_ready_df.randomSplit([0.8, 0.2], seed=42)

        mlflow.log_param("model_type", model_name)
        evaluator_auc = BinaryClassificationEvaluator(labelCol="Survived")

        if model_name == "lr":
            lr = LogisticRegression(featuresCol="features", labelCol="Survived")
            paramGrid = (
                ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .build()
            )
            cv = CrossValidator(
                estimator=lr,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator_auc,
                numFolds=3,
            )
            print("Starting hyperparameter tuning for Logistic Regression...")
            cvModel = cv.fit(train_data)
            model = cvModel.bestModel
            best_params = model.extractParamMap()
            mlflow.log_param("best_regParam", best_params[lr.regParam])
            mlflow.log_param("best_elasticNetParam", best_params[lr.elasticNetParam])

        elif model_name == "rf":
            rf = RandomForestClassifier(featuresCol="features", labelCol="Survived")
            paramGrid = (
                ParamGridBuilder()
                .addGrid(rf.numTrees, [50, 100, 150])
                .addGrid(rf.maxDepth, [5, 10, 15])
                .build()
            )
            cv = CrossValidator(
                estimator=rf,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator_auc,
                numFolds=3,
            )
            print("Starting hyperparameter tuning for Random Forest...")
            cvModel = cv.fit(train_data)
            model = cvModel.bestModel
            best_params = model.extractParamMap()
            mlflow.log_param("best_num_trees", best_params[rf.numTrees])
            mlflow.log_param("best_max_depth", best_params[rf.maxDepth])

        elif model_name == "dt":
            dt = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")
            paramGrid = (
                ParamGridBuilder()
                .addGrid(dt.maxDepth, [5, 10, 15])
                .addGrid(dt.maxBins, [32, 48])
                .build()
            )
            cv = CrossValidator(
                estimator=dt,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator_auc,
                numFolds=3,
            )
            print("Starting hyperparameter tuning for Decision Tree...")
            cvModel = cv.fit(train_data)
            model = cvModel.bestModel
            best_params = model.extractParamMap()
            mlflow.log_param("best_maxDepth", best_params[dt.maxDepth])
            mlflow.log_param("best_maxBins", best_params[dt.maxBins])

        else:
            spark.stop()
            raise ValueError("Unsupported model type specified.")

        print(f"Model training complete for {model_name}.")

        # --- Evaluation and Artifact Logging ---
        predictions = model.transform(test_data)
        auc = evaluator_auc.evaluate(predictions)
        mlflow.log_metric("test_auc", auc)
        print(f"AUC for {model_name}: {auc}")

        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="Survived", metricName="accuracy"
        )
        accuracy = evaluator_acc.evaluate(predictions)
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Accuracy on test data: {accuracy}")

        # --- Log Confusion Matrix ---
        preds_and_labels = predictions.select("prediction", "Survived").toPandas()
        confusion_matrix = pd.crosstab(
            preds_and_labels["Survived"], preds_and_labels["prediction"]
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png", "plots")
        print("Confusion matrix plot logged as an artifact.")

        # --- Log Model without Signature ---
        registered_model_name = f"TitanicClassifier_{model_name}"
        mlflow.spark.log_model(
            model,
            "spark-model",
            registered_model_name=registered_model_name,
        )
        print(f"Model logged and registered as '{registered_model_name}'.")

        spark.stop()

        # --- Promote Best Model ---
        promote_best_model(registered_model_name, "test_auc", run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="lr",
        choices=["lr", "rf", "dt"],
        help="Specify the model to train: lr, rf, or dt.",
    )
    args = parser.parse_args()
    train_model(model_name=args.model_name)