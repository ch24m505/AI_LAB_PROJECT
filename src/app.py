import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel, VectorAssembler

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define the input data schema using Pydantic
class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    FamilySize: int
    IsAlone: int
    Title: str

# Initialize the FastAPI app
app = FastAPI()

# --- Spark and Model Loading ---
# Create a SparkSession at startup
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# Load the trained model from the MLflow Model Registry
model_name = "TitanicClassifier_gbt" # Or whichever model is best
model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")


@app.get("/")
def read_root():
    """A simple endpoint to test if the API is running."""
    return {"message": f"Titanic Survival Prediction API is running. Model: {model_name}, Stage: {model_stage}."}


@app.post("/predict")
def predict_survival(passenger: PassengerFeatures):
    """
    Endpoint to predict passenger survival.
    Accepts passenger data, preprocesses it, and returns a survival prediction.
    """
    # Convert the input Pydantic object to a pandas DataFrame
    pdf = pd.DataFrame([passenger.dict()])
    
    # Convert the pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(pdf)

    # --- Recreate the *exact same* preprocessing pipeline ---
    # This must match the pipeline used in your `process_data.py`
    
    # Define categorical and numerical columns
    categorical_cols = ['Pclass', 'Sex', 'Embarked', 'Title']
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    
    # StringIndexer for categorical features
    indexers = [StringIndexerModel.from_labels([str(val) for val in spark_df.select(c).distinct().rdd.flatMap(lambda x: x).collect()], inputCol=c, outputCol=f"{c}_index") for c in categorical_cols]
    
    # OneHotEncoder for indexed categorical features
    encoders = [OneHotEncoderModel.from_dummy_values(range(spark_df.select(c).distinct().count()), inputCol=f"{c}_index", outputCol=f"{c}_ohe") for c in categorical_cols]
    
    # VectorAssembler to combine all features
    assembler_inputs = [f"{c}_ohe" for c in categorical_cols] + numerical_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Manually transform the data
    for indexer in indexers:
      spark_df = indexer.transform(spark_df)

    for encoder in encoders:
      spark_df = encoder.transform(spark_df)
      
    transformed_df = assembler.transform(spark_df)
    
    # Get the prediction from the loaded model using the transformed data
    prediction_result = model.predict(transformed_df.select("features"))
    
    # Extract the prediction
    prediction = int(prediction_result[0])
    
    # Return the prediction in a JSON response
    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Not Survived"
    }