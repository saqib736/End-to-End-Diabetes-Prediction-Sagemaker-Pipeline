# src/train.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # SageMaker automatically downloads the data to this path
    train_data_path = os.path.join('/opt/ml/input/data/train', 'train_data.csv')
    model_output_path = os.path.join('/opt/ml/model', 'model.joblib')

    # Load training data
    train_df = pd.read_csv(train_data_path)
    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']

    # Train the model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model to the output directory
    joblib.dump(model, model_output_path)
