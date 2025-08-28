# src/prepare_data.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def prepare_data():
    """
    Loads raw data, performs initial 'Embarked' imputation, splits the data,
    and saves the train and test sets as Parquet files.
    """
    spark = SparkSession.builder.appName("DataPreparation").getOrCreate()

    # Load raw data
    df = spark.read.csv("raw_dataset/train.csv", header=True, inferSchema=True)

    # Impute 'Embarked' with the mode before splitting
    mode_embarked = df.groupBy("Embarked").count().orderBy(col("count").desc()).first()[0]
    df = df.fillna(mode_embarked, subset=["Embarked"])

    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Save the splits
    output_dir = "data/prepared"
    os.makedirs(output_dir, exist_ok=True)
    train_df.write.mode("overwrite").parquet(os.path.join(output_dir, "train.parquet"))
    test_df.write.mode("overwrite").parquet(os.path.join(output_dir, "test.parquet"))
    
    print("Train and test sets saved to data/prepared/")
    spark.stop()

if __name__ == "__main__":
    prepare_data()