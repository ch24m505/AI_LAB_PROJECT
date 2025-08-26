import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data schema using Pydantic
class PassengerFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    # Add the new features you engineered
    FamilySize: int
    IsAlone: int
    Title: str

# Initialize the FastAPI app
app = FastAPI()

# --- Model Loading ---
# Load the model from the MLflow Model Registry at startup.
# This makes sure the model is ready to serve requests immediately.
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
    Accepts passenger data and returns a survival prediction.
    """
    # Convert the input Pydantic object to a dictionary
    data = passenger.dict()
    
    # Convert the dictionary to a pandas DataFrame
    # The model expects a DataFrame as input
    pdf = pd.DataFrame([data])
    
    # Get the prediction from the loaded model
    # The result is usually a numpy array or a pandas Series
    prediction_result = model.predict(pdf)
    
    # Extract the prediction (it's often the first element)
    prediction = int(prediction_result[0])
    
    # Return the prediction in a JSON response
    return {
        "prediction": prediction,
        "prediction_label": "Survived" if prediction == 1 else "Not Survived"
    }