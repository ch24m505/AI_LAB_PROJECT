import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data schema using Pydantic.
# This must match the raw data format.
class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    # Note: We don't need FamilySize or IsAlone here,
    # as the pipeline calculates them from SibSp and Parch.

# Initialize the FastAPI app
app = FastAPI()

# Load the full pipeline model from the MLflow Model Registry at startup.
# This single object contains all preprocessing and the trained classifier.
model_name = "TitanicSurvivalPipeline"
model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")


@app.get("/")
def read_root():
    """A simple endpoint to test if the API is running."""
    return {"message": f"Titanic Survival Prediction API is running. Model: {model_name}."}


@app.post("/predict")
def predict_survival(passenger: PassengerFeatures):
    """
    Accepts raw passenger data, uses the full pipeline to preprocess
    and predict, and returns the result.
    """
    # Convert the input Pydantic object to a pandas DataFrame
    pdf = pd.DataFrame([passenger.dict()])
    
    # The loaded model is a full pipeline. It will handle all preprocessing internally.
    prediction_result = model.predict(pdf)
    
    # Extract the prediction (it's often the first element of the result)
    prediction = int(prediction_result[0])
    
    # Return the prediction in a JSON response
    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Not Survived"
    }