import requests
import json

# The URL of your local API endpoint
URL = "http://127.0.0.1:8000/predict"

# Sample passenger data that matches the Pydantic schema in app.py
# This represents a 25-year-old male from Southampton in 3rd class
passenger_data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 25.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
    "FamilySize": 1,
    "IsAlone": 1,
    "Title": "Mr"
}

# Send a POST request to the API
try:
    response = requests.post(URL, data=json.dumps(passenger_data))
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    
    # Print the server's response
    print("API Request Sent Successfully!")
    print(f"Passenger Data Sent: {passenger_data}")
    print("-" * 30)
    print(f"API Response: {response.json()}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred while calling the API: {e}")