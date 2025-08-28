import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel, DecisionTreeClassificationModel, LogisticRegressionModel

# Define the input data schema
class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Initialize the FastAPI app
app = FastAPI()

# --- Load Spark and Models at Startup ---
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# Load the fitted preprocessing pipeline
preprocessor = PipelineModel.load("models/preprocessor")

# Load the trained classifier model
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

@app.post("/predict")
def predict_survival(passenger: PassengerFeatures):
    """
    Accepts raw passenger data, uses the SparkML preprocessor and model
    to predict, and returns the result.
    """
    # Convert input to a Spark DataFrame
    pdf = pd.DataFrame([passenger.dict()])
    spark_df = spark.createDataFrame(pdf)
    
    # Preprocess the data using the loaded pipeline
    processed_df = preprocessor.transform(spark_df)
    
    # Make a prediction using the loaded classifier
    prediction_df = classifier.transform(processed_df)
    
    # Extract the prediction result
    prediction = prediction_df.select("prediction").first()['prediction']
    
    return {
        "prediction": int(prediction),
        "prediction_label": "Survived" if prediction == 1 else "Not Survived"
    }