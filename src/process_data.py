#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, mean, mode

def create_preprocessing_pipeline(df):
    """
    Creates a PySpark ML Pipeline for preprocessing the Titanic dataset.

    Args:
        df: The raw Spark DataFrame.

    Returns:
        A PySpark ML Pipeline model.
    """

    # 1. Handle Missing Values
    # Impute missing 'Age' with the mean.
    mean_age = df.agg(mean(col('Age'))).collect()[0][0]

    # Impute missing 'Embarked' with the mode.
    mode_embarked = df.groupBy('Embarked').count().orderBy(col('count').desc()).first()[0]

    # Impute missing 'Fare' with the mean.
    mean_fare = df.agg(mean(col('Fare'))).collect()[0][0]

    # Use a custom transformer or direct operations for imputation.
    # For this example, we'll apply these operations before the pipeline.
    # In a real-world scenario, you might use a custom Transformer.
    df = df.fillna(mean_age, subset=['Age'])
    df = df.fillna(mode_embarked, subset=['Embarked'])
    df = df.fillna(mean_fare, subset=['Fare'])

    # 2. Feature Engineering
    df = df.withColumn('FamilySize', col('SibSp') + col('Parch') + 1)
    df = df.withColumn('IsAlone', when(col('FamilySize') == 1, 1).otherwise(0))

    # 3. Define Categorical and Numerical Columns
    categorical_cols = ['Pclass', 'Sex', 'Embarked']
    numerical_cols = ['Age', 'Fare', 'FamilySize', 'IsAlone']

    # 4. Create Pipeline Stages

    # Index all categorical columns
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid='keep')
        for c in categorical_cols
    ]

    # One-hot encode the indexed columns
    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vector")
        for c in categorical_cols
    ]

    # Final list of all features to be assembled
    feature_columns = numerical_cols + [f"{c}_vector" for c in categorical_cols]

    # Assemble all features into a single vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # 5. Create the Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    return pipeline, df


def process_and_save_data():
    """
    Main function to run the data processing pipeline.
    """
    spark = SparkSession.builder \
        .appName("TitanicMLPreprocessing") \
        .getOrCreate()

    try:
        df = spark.read.csv('C:/Users/user/AI_LAB_PROJECT/data/raw/train.csv', header=True, inferSchema=True)
        print("Raw training data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        spark.stop()
        return

    # Create and fit the pipeline
    pipeline, df_imputed = create_preprocessing_pipeline(df)
    pipeline_model = pipeline.fit(df_imputed)

    # Transform the data using the fitted pipeline model
    processed_df = pipeline_model.transform(df_imputed)

    # Select the final columns: 'Survived' and the 'features' vector
    final_df = processed_df.select('Survived', 'features')

    print("Pipeline successfully transformed data.")
    final_df.printSchema()
    final_df.show(5, truncate=False)

    try:
        # Save the processed DataFrame. This can now be used for model training.
        final_df.write.mode('overwrite').parquet('../data/processed/titanic_ml_ready')
        print("Processed data saved to 'data/processed/titanic_ml_ready'.")
    except Exception as e:
        print(f"Error saving processed data: {e}")

    spark.stop()

if __name__ == '__main__':
    process_and_save_data()


# In[ ]:




