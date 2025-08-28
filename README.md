============================================== MLOps Pipeline for Titanic Survival Prediction
This project implements a robust, end-to-end MLOps pipeline for a classification task using the Kaggle Titanic dataset. The pipeline is built with a modular, multi-stage architecture using DVC and leverages SparkML for distributed processing and training, MLflow for experiment tracking, and Docker for containerized deployment.

Technologies Used
Data Processing & Modeling: Apache Spark (PySpark), SparkML

Pipeline & Data Versioning: DVC (Data Version Control)

Experiment Tracking: MLflow

API Deployment: FastAPI, Uvicorn

Containerization: Docker

Environment Management: Conda

Setup Instructions
Follow these steps to set up the project environment and download the necessary data.

Prerequisites:

Conda

Git

Docker Desktop

Clone the Repository
'git clone https://github.com/ch24m505/AI_LAB_PROJECT.git'
'cd AI_LAB_PROJECT'

Create and Activate the Conda Environment
'conda create --name ai_lab python=3.12 -y'
'conda activate ai_lab'
'pip install -r requirements.txt'

Pull the Data
'dvc pull'

Running the Pipeline
Reproduce the Full Pipeline
To run all stages, use the 'dvc repro' command.
'dvc repro'

View Experiment Results
Start the MLflow Server in a dedicated terminal:
'mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000'
Then, open your browser to 'http://127.0.0.1:5000'.

Deploy and Test the API
Build the Docker Image:
'docker build -t titanic-api .'

Run the Docker Container in a dedicated terminal:
'docker run -p 8000:8000 titanic-api'

Test the API in a new terminal:
'conda activate ai_lab'
'python3 src/test_api.py'
You can also visit the interactive docs at 'http://127.0.0.1:8000/docs'.

Project Structure
.
├── data/
│   └── prepared/         (Processed train/test splits)
├── models/
│   ├── preprocessor/     (Fitted SparkML preprocessor)
│   └── classifier/       (Final trained SparkML classifier)
├── raw_dataset/
│   └── train.csv         (Raw data tracked by DVC)
├── reports/
│   └── drift_report.json (Drift detection output)
├── src/
│   ├── prepare_data.py   (Stage 1: Data splitting)
│   ├── preprocess.py     (Stage 2: Preprocessor creation)
│   ├── train.py          (Stage 3: Model training and selection)
│   ├── evaluate.py       (Stage 4: Final model evaluation)
│   ├── detect_drift.py   (Stage 5: Drift detection)
│   └── fast_api_app.py   (FastAPI application for deployment)
├── .gitignore
├── dvc.yaml              (DVC pipeline definition)
├── Dockerfile            (Docker instructions for the API)
└── requirements.txt      (Project dependencies)