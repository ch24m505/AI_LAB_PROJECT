import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier,
                                       GBTClassifier, LogisticRegression,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from sklearn.metrics import confusion_matrix

import mlflow
import mlflow.spark


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save the plot to a file
    plot_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def main(model_name):
    """Main function for training and evaluating a model."""
    spark = SparkSession.builder \
        .appName("TitanicMLOps") \
        .getOrCreate()

    # --- Load Data ---
    processed_df = spark.read.parquet("data/processed/titanic_ml_ready")
    train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

    # --- MLflow Setup ---
    mlflow.set_experiment("Titanic Survival Prediction")

    # --- Model Selection and Hyperparameter Grid ---
    if model_name == 'lr':
        lr = LogisticRegression(featuresCol='features', labelCol='Survived')
        paramGrid = (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                     .build())
        model = lr
        mlflow_run_name = "lr_hyperparam_tuning"
        model_registry_name = "TitanicClassifier_lr"

    elif model_name == 'dt':
        dt = DecisionTreeClassifier(featuresCol='features', labelCol='Survived')
        paramGrid = (ParamGridBuilder()
                     .addGrid(dt.maxDepth, [3, 5, 7])
                     .addGrid(dt.maxBins, [32, 50])
                     .build())
        model = dt
        mlflow_run_name = "dt_hyperparam_tuning"
        model_registry_name = "TitanicClassifier_dt"

    elif model_name == 'rf':
        rf = RandomForestClassifier(featuresCol='features', labelCol='Survived')
        paramGrid = (ParamGridBuilder()
                     .addGrid(rf.numTrees, [50, 100, 150])
                     .addGrid(rf.maxDepth, [5, 8, 10])
                     .addGrid(rf.maxBins, [32, 50])
                     .build())
        model = rf
        mlflow_run_name = "rf_hyperparam_tuning"
        model_registry_name = "TitanicClassifier_rf"
    
    # --- NEW: GBT Classifier ---
    elif model_name == 'gbt':
        gbt = GBTClassifier(featuresCol='features', labelCol='Survived')
        paramGrid = (ParamGridBuilder()
                     .addGrid(gbt.maxIter, [10, 20, 30])
                     .addGrid(gbt.maxDepth, [3, 5])
                     .build())
        model = gbt
        mlflow_run_name = "gbt_hyperparam_tuning"
        model_registry_name = "TitanicClassifier_gbt"

    else:
        raise ValueError("Invalid model_name. Choose from 'lr', 'dt', 'rf', 'gbt'.")

    # --- Start MLflow Run ---
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # --- Cross-Validation and Training ---
        evaluator = BinaryClassificationEvaluator(labelCol="Survived")
        
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=5)

        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel

        # --- Evaluation ---
        predictions = best_model.transform(test_df)
        test_auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        
        # Calculate accuracy manually for logging
        correct_predictions = predictions.filter(predictions.Survived == predictions.prediction).count()
        total_data = predictions.count()
        test_accuracy = correct_predictions / total_data

        print(f"Model: {model_name.upper()}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # --- Logging to MLflow ---
        mlflow.log_param("model_type", model_name)
        
        # Log best hyperparameters
        best_params = best_model.extractParamMap()
        for param, value in best_params.items():
            if any(p.name == param.name for p in paramGrid[0]):
                mlflow.log_param(param.name, value)
        
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log confusion matrix artifact
        preds_for_cm = predictions.select("Survived", "prediction").toPandas()
        cm_path = plot_confusion_matrix(preds_for_cm["Survived"], preds_for_cm["prediction"], model_name)
        mlflow.log_artifact(cm_path, "plots")
        os.remove(cm_path)

        # Log and register the model
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="model",
            registered_model_name=model_registry_name
        )

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="rf",
                        choices=["lr", "dt", "rf", "gbt"],
                        help="Choose the model to train.")
    args = parser.parse_args()
    main(args.model_name)