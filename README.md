==============================================
MLOps Pipeline for Titanic Survival Prediction
==============================================

This project implements a robust, end-to-end MLOps pipeline for a classification task using the Kaggle Titanic dataset. The pipeline is built with a modular, multi-stage architecture using DVC and leverages SparkML for distributed processing and training, MLflow for experiment tracking, and Docker for containerized deployment.

---
Technologies Used
---

- Data Processing & Modeling: Apache Spark (PySpark), SparkML
- Pipeline & Data Versioning: DVC (Data Version Control)
- Experiment Tracking: MLflow
- API Deployment: FastAPI, Uvicorn
- Containerization: Docker
- Environment Management: Conda

---
Setup and Execution Guide
---

Follow these steps to set up the environment, reproduce the pipeline, and deploy the final model.

1. Prerequisites
   - Conda
   - Git
   - Docker Desktop

2. Clone the Repository
   git clone https://github.com/ch24m505/AI_LAB_PROJECT.git
   cd AI_LAB_PROJECT

3. Set Up the Environment
   # Create and activate the Conda environment
   conda create --name ai_lab python=3.12 -y
   conda activate ai_lab

   # Install all required packages
   pip install -r requirements.txt

4. Place the Dataset
   This project uses DVC to track the raw dataset, but for local evaluation, you will need to manually place the data file.
   1. Download the Titanic dataset from Kaggle: https://www.kaggle.com/c/titanic/data
   2. Copy 'train.csv' into the 'raw_dataset' folder in the project's root directory(AI_LAB_PROJECT).

---
Running the Pipeline
---

The DVC pipeline requires a live MLflow server to log experiments. This requires two terminals.

Step 1: Start the MLflow Server (in Terminal 1)
   This server must remain running while you execute the pipeline.

   # Navigate to the project directory (AI_LAB_PROJECT) and activate the environment
   conda activate ai_lab

   # Start the server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   
   (You can view the MLflow UI in your browser at http://127.0.0.1:5000)

Step 2: Reproduce the Full Pipeline (in Terminal 2)
   In a new terminal, run the 'dvc repro' command. This will execute all stages.

   # Navigate to the project directory (AI_LAB_PROJECT) and activate the environment
   conda activate ai_lab

   # Run all pipeline stages
   dvc repro

---
Deploy and Test the API
---

Step 1: Build the Docker Image
   docker build -t titanic-api .

Step 2: Run the Docker Container (in Terminal 1)
   This starts the live API server.
   docker run -p 8000:8000 titanic-api

Step 3: Test the API (in Terminal 2)
   # Navigate to the project directory (AI_LAB_PROJECT) and Activate the environment
   conda activate ai_lab

   # Run the test script
   python3 src/test_api.py
   
   (You can also interact with the API by visiting the auto-generated documentation at http://127.0.0.1:8000/docs)

---
Project Structure
---

.
├── data/
│   └── prepared/         (Processed train/test splits)
├── models/
│   ├── preprocessor/     (Fitted SparkML preprocessor)
│   └── classifier/       (Final trained SparkML classifier)
├── raw_dataset/
│   └── train.csv         (Raw data)
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