import requests
import json

# The URL of your local API endpoint
URL = "http://127.0.0.1:8000/predict"

# Sample passenger data that matches the raw input format.
# This represents a 30-year-old female from Cherbourg in 1st class.
passenger_data = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 30.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 80.0,
    "Embarked": "C"
}

# Send a POST request to the API
try:
    response = requests.post(URL, data=json.dumps(passenger_data))
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    
    # Print the server's response
    print("API Request Sent Successfully!")
    print(f"Passenger Data Sent: {passenger_data}")
    print("-" * 30)
    print(f"API Response: {response.json()}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred while calling the API: {e}")