# src/test_api.py
import requests
import json

# The URL of your API running in the Docker container
URL = "http://127.0.0.1:8000/predict"

# Sample passenger data (a 1st class female who should survive)
passenger_data = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 38.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Embarked": "C"
}

print("Sending request to the API...")
try:
    # Send the request
    response = requests.post(URL, data=json.dumps(passenger_data))
    response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)

    # Print the successful response from the server
    print("Request successful!")
    print("-" * 30)
    print(f"Data Sent: {passenger_data}")
    print(f"Prediction Received: {response.json()}")
    print("-" * 30)

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")