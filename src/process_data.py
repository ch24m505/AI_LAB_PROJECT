#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, count

def process_data():
    """
    This function loads the raw Titanic dataset, performs all the necessary
    preprocessing and feature engineering, and saves the cleaned dataset.
    """
    # 1. Initialize Spark Session
    # This is the entry point to any Spark functionality.
    spark = SparkSession.builder \
        .appName("TitanicDataProcessing") \
        .getOrCreate()
    print("Spark session initialized.")

    # 2. Load the Raw Data
    # Reading the training data from the 'data/raw' directory.
    # 'header=True' treats the first row as column names.
    # 'inferSchema=True' automatically detects column data types.
    try:
        df = spark.read.csv('../data/raw/train.csv', header=True, inferSchema=True)
        print("Raw training data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        spark.stop()
        return

    # 3. Data Cleaning and Feature Engineering

    # --- Handle Missing Values ---

    # Fill missing 'Age' values with the mean age of the dataset.
    mean_age = df.select(mean(col('Age'))).collect()[0][0]
    df = df.fillna(mean_age, subset=['Age'])
    print(f"Missing 'Age' values filled with mean age: {mean_age:.2f}")

    # Fill missing 'Embarked' values with the most frequent value ('S').
    # In a more advanced pipeline, you would calculate this dynamically.
    df = df.fillna('S', subset=['Embarked'])
    print("Missing 'Embarked' values filled with 'S'.")

    # --- Create New Features ---

    # Create 'FamilySize' from 'SibSp' (siblings/spouses) and 'Parch' (parents/children).
    df = df.withColumn('FamilySize', col('SibSp') + col('Parch') + 1)
    print("Created 'FamilySize' feature.")

    # --- Convert Categorical Features to Numerical ---

    # Convert 'Sex' to a binary numeric column: male=1, female=0.
    df = df.withColumn('Sex_numeric', when(col('Sex') == 'male', 1).otherwise(0))
    print("Converted 'Sex' to numeric format.")

    # Convert 'Embarked' to numerical format using one-hot encoding principles.
    df = df.withColumn('Embarked_S', when(col('Embarked') == 'S', 1).otherwise(0))
    df = df.withColumn('Embarked_C', when(col('Embarked') == 'C', 1).otherwise(0))
    df = df.withColumn('Embarked_Q', when(col('Embarked') == 'Q', 1).otherwise(0))
    print("Converted 'Embarked' to numeric format.")

    # --- Select and Drop Columns ---

    # Select the columns useful for machine learning and drop the original/unneeded ones.
    final_df = df.select(
        'Survived',
        'Pclass',
        'Age',
        'FamilySize',
        'Fare',
        'Sex_numeric',
        'Embarked_S',
        'Embarked_C',
        'Embarked_Q'
    )

    print("Final features selected.")
    final_df.printSchema()
    final_df.show(10)

    # 4. Save the Processed Data
    # Saving the final DataFrame in Parquet format for efficient storage and retrieval.
    # 'mode('overwrite')' allows the script to be re-run without errors.
    try:
        final_df.write.mode('overwrite').parquet('../data/processed/titanic_processed')
        print("Processed data successfully saved to 'data/processed/titanic_processed'.")
    except Exception as e:
        print(f"Error saving processed data: {e}")

    # 5. Stop the Spark Session
    # It's important to stop the session to free up resources.
    spark.stop()
    print("Spark session stopped.")

if __name__ == '__main__':
    process_data()


# In[ ]:




