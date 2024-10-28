import joblib
import os
import numpy as np
import json

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Deserialize the request body."""
    if request_content_type == 'application/json':
        # Parse the JSON input
        input_data = json.loads(request_body)
        # Define the feature order
        feature_order = ["Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
                         "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability",
                         "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"]
        if isinstance(input_data, dict):
            # Extract features in the defined order
            input_data = [input_data[feature] for feature in feature_order]
        elif isinstance(input_data, list):
            # Handle batch input
            input_data = [[item[feature] for feature in feature_order] for item in input_data]
        else:
            raise ValueError("Invalid input format. Expected dict or list.")
        # Convert to numpy array
        data = np.array(input_data)
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions using the model."""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    """Serialize the prediction output."""
    if response_content_type == 'application/json':
        # Convert prediction to a list and then to JSON
        result = prediction.tolist()
        return json.dumps(result), response_content_type
    elif response_content_type == 'text/csv':
        # Convert prediction to CSV format
        result = ','.join(map(str, prediction))
        return result, response_content_type
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
