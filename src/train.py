# src/train.py
import os
import mlflow
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_and_select_model():
    """
    Trains and tunes multiple SparkML models, logs results to MLflow,
    and saves the best performing model.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

    # Load preprocessor and training data
    preprocessor = PipelineModel.load("models/preprocessor")
    train_df = spark.read.parquet("data/prepared/train.parquet")
    processed_train_df = preprocessor.transform(train_df)

    # Define models and their hyperparameter grids
    lr = LogisticRegression(featuresCol="features", labelCol="Survived")
    lr_grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0])          # regularization strength
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])           # 0=L2, 1=L1, 0.5=ElasticNet
    .addGrid(lr.maxIter, [50, 100])                         # number of iterations
    .build())

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Survived")
    dt_grid = (
    ParamGridBuilder()
    .addGrid(dt.maxDepth, [3, 5, 10, 15])                   # tree depth
    .addGrid(dt.minInstancesPerNode, [1, 2, 5])             # min samples per leaf
    .addGrid(dt.impurity, ["gini", "entropy"])              # splitting criterion
    .build())

    rf = RandomForestClassifier(featuresCol="features", labelCol="Survived")
    rf_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100])                   # number of trees
    .addGrid(rf.maxDepth, [5, 10])                      # max depth
    .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"])    # feature sampling
    .addGrid(rf.minInstancesPerNode, [1, 2, 4])             # min leaf size
    .build())

    gbt = GBTClassifier(featuresCol="features", labelCol="Survived")
    gbt_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxIter, [50, 25])                        # boosting rounds
    .addGrid(gbt.maxDepth, [3, 5])                       # tree depth
    .addGrid(gbt.stepSize, [0.05, 0.1])                # learning rate
    .build())
    
    models = {
        "LogisticRegression": (lr, lr_grid),
        "DecisionTree": (dt, dt_grid),
        "RandomForest": (rf, rf_grid),
        "GBT": (gbt, gbt_grid)
    }

    best_model = None
    best_cv_score = -1.0
    
    evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")

    with mlflow.start_run(run_name="Automated SparkML Model Selection") as parent_run:
        for model_name, (model, param_grid) in models.items():
            with mlflow.start_run(run_name=f"Tune_{model_name}", nested=True) as child_run:
                cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
                
                print(f"Starting tuning for {model_name}...")
                cv_model = cv.fit(processed_train_df)
                
                avg_cv_score = max(cv_model.avgMetrics)
                mlflow.log_metric("avg_cv_auc", avg_cv_score)
                
                # Log best params
                best_params = {p.name: v for p, v in cv_model.bestModel.extractParamMap().items() if p in param_grid[0]}
                mlflow.log_params(best_params)
                
                if avg_cv_score > best_cv_score:
                    best_cv_score = avg_cv_score
                    best_model = cv_model.bestModel
                    mlflow.set_tag("best_model_type", model_name)

        mlflow.log_metric("best_overall_cv_auc", best_cv_score)
        
        # Save the single best classifier model
        output_path = "models/classifier"
        os.makedirs("models", exist_ok=True)
        best_model.write().overwrite().save(output_path)
        print(f"\nBest model ({best_model.__class__.__name__}) saved to {output_path}")

        # Register the best model in the MLflow Model Registry
        # This makes it appear in the "Models" tab
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="champion-classifier",
            registered_model_name="TitanicChampionClassifier"
        )
        print("Best model has been registered in the MLflow Model Registry.")


    spark.stop()

if __name__ == "__main__":
    train_and_select_model()