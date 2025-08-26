import argparse
import mlflow
import mlflow.spark
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
    SQLTransformer,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    DecisionTreeClassifier,
    GBTClassifier,
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
        print(f"Metric '{metric_name}' not found. Skipping promotion.")
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

    print(f"Current model metric: {current_metric:.4f}, Production: {production_metric:.4f}")

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


def train_pipeline(model_name: str):
    """
    Runs the complete, end-to-end pipeline including all preprocessing and training logic.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name=f"{model_name}_full_pipeline_run") as run:
        spark = SparkSession.builder.appName("TitanicFullPipeline").getOrCreate()

        raw_df = spark.read.csv("raw_dataset/train.csv", header=True, inferSchema=True)
        mode_embarked = raw_df.groupBy("Embarked").count().orderBy(col("count").desc()).first()[0]
        raw_df = raw_df.fillna(mode_embarked, subset=["Embarked"])
        train_data, test_data = raw_df.randomSplit([0.8, 0.2], seed=42)

        # --- Define ALL Preprocessing Stages ---
        imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age_imputed", "Fare_imputed"]).setStrategy("mean")
        family_size_tf = SQLTransformer(statement="SELECT *, SibSp + Parch + 1 AS FamilySize FROM __THIS__")
        is_alone_tf = SQLTransformer(statement="SELECT *, CASE WHEN FamilySize = 1 THEN 1 ELSE 0 END AS IsAlone FROM __THIS__")
        categorical_cols = ["Pclass", "Sex", "Embarked"]
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
        encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_ohe") for c in categorical_cols]
        numerical_cols = ["Age_imputed", "Fare_imputed", "FamilySize", "IsAlone"]
        assembler_inputs = [f"{c}_ohe" for c in categorical_cols] + numerical_cols
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

        # --- Define Classifier and Hyperparameter Grid ---
        if model_name == "lr":
            classifier = LogisticRegression(featuresCol="features", labelCol="Survived")
            paramGrid = (ParamGridBuilder().addGrid(classifier.regParam, [0.01, 0.1, 0.5]).addGrid(classifier.elasticNetParam, [0.0, 0.5, 1.0]).build())
        elif model_name == "rf":
            classifier = RandomForestClassifier(featuresCol="features", labelCol="Survived")
            paramGrid = (ParamGridBuilder().addGrid(classifier.numTrees, [50, 100, 150]).addGrid(classifier.maxDepth, [5, 10, 15]).build())
        elif model_name == "dt":
            classifier = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")
            paramGrid = (ParamGridBuilder().addGrid(classifier.maxDepth, [5, 10, 15]).addGrid(classifier.maxBins, [32, 48]).build())
        elif model_name == "gbt":
            classifier = GBTClassifier(featuresCol="features", labelCol="Survived")
            paramGrid = (ParamGridBuilder().addGrid(classifier.maxIter, [10, 20]).addGrid(classifier.maxDepth, [3, 5]).build())
        else:
            raise ValueError("Unsupported model type specified.")

        # --- Create the Full Pipeline ---
        all_stages = [imputer, family_size_tf, is_alone_tf] + indexers + encoders + [assembler, classifier]
        pipeline = Pipeline(stages=all_stages)

        # --- Run Cross-Validation on the Full Pipeline ---
        evaluator_auc = BinaryClassificationEvaluator(labelCol="Survived")
        cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator_auc, numFolds=3)
        
        print(f"Starting tuning for the full {model_name} pipeline...")
        cv_model = cv.fit(train_data)
        
        # --- NEW: Log Best Hyperparameters ---
        best_model = cv_model.bestModel.stages[-1] # Get the classifier stage from the best pipeline
        for param in paramGrid[0]:
            param_name = param.name
            param_value = best_model.getOrDefault(param)
            mlflow.log_param(f"best_{param_name}", param_value)
            print(f"Logged best param: {param_name} = {param_value}")

        # --- Evaluate and Log Metrics ---
        predictions = cv_model.transform(test_data)
        auc = evaluator_auc.evaluate(predictions)
        mlflow.log_metric("test_auc", auc)
        
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="Survived", metricName="accuracy")
        accuracy = evaluator_acc.evaluate(predictions)
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Test AUC: {auc:.4f}, Test Accuracy: {accuracy:.4f}")

        # --- Log Artifacts and Model ---
        preds_and_labels = predictions.select("prediction", "Survived").toPandas()
        cm = pd.crosstab(preds_and_labels["Survived"], preds_and_labels["prediction"])
        plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt="d"); plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png"); mlflow.log_artifact("confusion_matrix.png", "plots")

        registered_model_name = "TitanicSurvivalPipeline"
        mlflow.spark.log_model(spark_model=cv_model.bestModel, artifact_path="model", registered_model_name=registered_model_name)
        print(f"Full pipeline model logged as '{registered_model_name}'.")
        
        # --- Promote Model if it's the Best ---
        promote_best_model(registered_model_name, "test_auc", run.info.run_id)

        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gbt", choices=["lr", "rf", "dt", "gbt"])
    args = parser.parse_args()
    train_pipeline(model_name=args.model_name)