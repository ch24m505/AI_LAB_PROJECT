#!/usr/bin/env python
# coding: utf-8

# In[7]:


# All necessary imports are included here
import mlflow
import mlflow.spark
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.tracking import MlflowClient

def train_model(model_name: str):
    """
    Loads ML-ready data, trains a selected model, and logs with MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        spark = SparkSession.builder.appName("TitanicModelTraining").getOrCreate()

        # Use the absolute path to avoid issues
        ml_ready_df = spark.read.parquet('C:/Users/user/AI_LAB_PROJECT/data/processed/titanic_ml_ready')
        train_data, test_data = ml_ready_df.randomSplit([0.8, 0.2], seed=42)

        # --- Model Selection and Training ---
        mlflow.log_param("model_type", model_name)

        if model_name == "lr":
            model_instance = LogisticRegression(featuresCol="features", labelCol="Survived")
            model = model_instance.fit(train_data)

        elif model_name == "rf":
            num_trees = 100
            mlflow.log_param("num_trees", num_trees)
            model_instance = RandomForestClassifier(featuresCol="features", labelCol="Survived", numTrees=num_trees)
            model = model_instance.fit(train_data)

        elif model_name == "dt":
            max_depth = 5
            mlflow.log_param("max_depth", max_depth)
            model_instance = DecisionTreeClassifier(featuresCol="features", labelCol="Survived", maxDepth=max_depth)
            model = model_instance.fit(train_data)

        else:
            raise ValueError("Unsupported model type specified.")

        print(f"Model training complete for {model_name}.")

        # --- Evaluation and Logging ---
        predictions = model.transform(test_data)
        evaluator_auc = BinaryClassificationEvaluator(labelCol="Survived")
        auc = evaluator_auc.evaluate(predictions)
        mlflow.log_metric("test_auc", auc)
        print(f"AUC for {model_name}: {auc}")

        # input_example = train_data.limit(1)

        mlflow.spark.log_model(
        model,
        "spark-model",
        registered_model_name=f"TitanicClassifier_{model_name}"
        )

        # Note: The promote_best_model function is not included here for simplicity,
        # but you can add it back if you want to test that logic as well.

        spark.stop()

# --- Main execution block for Jupyter Notebook ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="lr",
        choices=["lr", "rf", "dt"],
        help="Specify the model to train: lr, rf, or dt."
    )
    args = parser.parse_args()

    train_model(model_name=args.model_name)


# In[ ]:




