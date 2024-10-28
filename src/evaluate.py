# src/evaluate.py

import os
import json
import tarfile
import joblib
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model_path, test_data_path, report_path):
    # Extract the model artifact
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')

    # Load the model
    model = joblib.load('model.joblib')

    # Load test data
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('class', axis=1)
    y_test = test_df['class']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Save evaluation report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump({'metrics': metrics}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/model.tar.gz')
    parser.add_argument('--test_data_path', type=str, default='data/test_data.csv')
    parser.add_argument('--report_path', type=str, default='reports/evaluation.json')
    args = parser.parse_args()
    evaluate(args.model_path, args.test_data_path, args.report_path)
