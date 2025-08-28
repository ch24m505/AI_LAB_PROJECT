# src/preprocess.py
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer, SQLTransformer
)

def create_and_save_preprocessor():
    """
    Creates and fits a SparkML preprocessing pipeline on the training data,
    then saves the fitted pipeline model.
    """
    spark = SparkSession.builder.appName("PreprocessorCreation").getOrCreate()

    # Load only the training data to fit the preprocessor
    train_df = spark.read.parquet("data/prepared/train.parquet")

    # --- Define ALL Preprocessing Stages ---
    imputer = Imputer(inputCols=["Age", "Fare"], outputCols=["Age_imputed", "Fare_imputed"]).setStrategy("mean")
    family_size_tf = SQLTransformer(statement="SELECT *, SibSp + Parch + 1 AS FamilySize FROM __THIS__")
    is_alone_tf = SQLTransformer(statement="SELECT *, CASE WHEN FamilySize = 1 THEN 1 ELSE 0 END AS IsAlone FROM __THIS__")
    categorical_cols = ["Pclass", "Sex", "Embarked"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_ohe") for c in categorical_cols]
    numerical_cols = ["Age_imputed", "Fare_imputed", "FamilySize", "IsAlone"]
    assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in categorical_cols] + numerical_cols, outputCol="features")

    # --- Create and Fit the Preprocessing Pipeline ---
    preprocessing_pipeline = Pipeline(stages=[imputer, family_size_tf, is_alone_tf] + indexers + encoders + [assembler])
    
    print("Fitting the preprocessing pipeline...")
    preprocessor_model = preprocessing_pipeline.fit(train_df)
    
    # --- Save the Fitted Preprocessing Pipeline ---
    output_path = "models/preprocessor"
    os.makedirs("models", exist_ok=True)
    preprocessor_model.write().overwrite().save(output_path)
    
    print(f"Fitted preprocessing pipeline saved to {output_path}")
    spark.stop()

if __name__ == "__main__":
    create_and_save_preprocessor()