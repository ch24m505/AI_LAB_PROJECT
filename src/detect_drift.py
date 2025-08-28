# src/detect_drift.py
import os
import json
from pyspark.sql import SparkSession

def detect_drift():
    """
    Calculates and compares summary statistics for numerical and categorical
    features to detect data drift between training and new data.
    """
    spark = SparkSession.builder.appName("DriftDetection").getOrCreate()

    # Load baseline training data and a sample of new data
    train_df = spark.read.parquet("data/prepared/train.parquet")
    new_data_df = spark.read.parquet("data/prepared/test.parquet").limit(50)

    drift_report = {}
    overall_drift_detected = False

    # --- 1. Numerical Drift Detection (Comparing Summary Statistics) ---
    numerical_cols = ["Age", "Fare"]
    print("--- Running Numerical Drift Detection ---")

    # Get summary statistics as a dictionary
    train_stats = {row['summary']: row for row in train_df.describe(numerical_cols).collect()}
    new_stats = {row['summary']: row for row in new_data_df.describe(numerical_cols).collect()}

    for col_name in numerical_cols:
        train_mean = float(train_stats['mean'][col_name]) if train_stats['mean'][col_name] is not None else 0
        new_mean = float(new_stats['mean'][col_name]) if new_stats['mean'][col_name] is not None else 0
        
        drift_threshold = 0.15 # 15% threshold
        mean_drift = abs(new_mean - train_mean) / train_mean if train_mean != 0 else 0
        drift_detected = mean_drift > drift_threshold

        if drift_detected:
            overall_drift_detected = True
        
        print(f"Feature '{col_name}': Mean drift = {mean_drift:.2%}. Drift Detected: {drift_detected}")
        drift_report[f"{col_name}_drift"] = {
            'test': 'Mean Shift',
            'baseline_mean': train_mean,
            'new_data_mean': new_mean,
            'drift_detected': drift_detected
        }

    # --- 2. Categorical Drift Detection (Unseen Category Check) ---
    print("\n--- Running Categorical Drift Detection ---")
    baseline_categories = {row['Embarked'] for row in train_df.select('Embarked').distinct().collect()}
    new_categories = {row['Embarked'] for row in new_data_df.select('Embarked').distinct().collect()}
    unseen_categories = new_categories - baseline_categories
    
    categorical_drift_detected = bool(unseen_categories)
    if categorical_drift_detected:
        overall_drift_detected = True

    print(f"Feature 'Embarked': Unseen categories = {unseen_categories or 'None'}. Drift Detected: {categorical_drift_detected}")
    drift_report["Embarked_drift"] = {
        'test': 'Unseen Category Check',
        'unseen_values': list(unseen_categories),
        'drift_detected': categorical_drift_detected
    }
    
    # --- 3. Generate Final Report ---
    os.makedirs("reports", exist_ok=True)
    # Corrected line below
    with open("reports/drift_report.json", "w") as f:
        json.dump(drift_report, f, indent=4)
    
    print("\nDrift report generated at reports/drift_report.json")

    if overall_drift_detected:
        print("\nALERT: Significant data drift detected in one or more features!")
    else:
        print("\nNo significant drift detected.")
        
    spark.stop()

if __name__ == "__main__":
    detect_drift()