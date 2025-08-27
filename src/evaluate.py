# src/evaluate.py
import json
import os
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel, DecisionTreeClassificationModel, LogisticRegressionModel
# --- THIS LINE WAS ADDED ---
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def evaluate_model():
    """
    Loads the preprocessor and the champion classifier, evaluates on test data,
    and saves the final metrics.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    spark = SparkSession.builder.appName("ModelEvaluation").getOrCreate()

    # Load artifacts
    preprocessor = PipelineModel.load("models/preprocessor")
    
    model_path = "models/classifier"
    try:
        classifier = GBTClassificationModel.load(model_path)
    except Exception:
        try:
            classifier = RandomForestClassificationModel.load(model_path)
        except Exception:
            try:
                classifier = DecisionTreeClassificationModel.load(model_path)
            except Exception:
                classifier = LogisticRegressionModel.load(model_path)

    test_df = spark.read.parquet("data/prepared/test.parquet")
    
    # Apply transformations and make predictions
    processed_test_df = preprocessor.transform(test_df)
    predictions = classifier.transform(processed_test_df)

    # Calculate and log metrics
    auc = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC").evaluate(predictions)
    
    # Using MulticlassClassificationEvaluator for accuracy is more direct in Spark
    accuracy = MulticlassClassificationEvaluator(labelCol="Survived", metricName="accuracy").evaluate(predictions)
    
    print(f"Final Test AUC: {auc:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")

    metrics = {"final_test_auc": auc, "final_test_accuracy": accuracy}
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    mlflow.log_metrics(metrics)
    spark.stop()

if __name__ == "__main__":
    evaluate_model()