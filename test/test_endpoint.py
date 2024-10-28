import boto3
import json

# Initialize the SageMaker runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Endpoint name
endpoint_name = 'DiabetesPredictionEndpoint'  # Replace with your endpoint name

# Input data as a JSON object
input_data = {
    "Age": 0.5,
    "Gender": 1,
    "Polyuria": 0,
    "Polydipsia": 0,
    "sudden weight loss": 0,
    "weakness": 0,
    "Polyphagia": 0,
    "Genital thrush": 1,
    "visual blurring": 0,
    "Itching": 1,
    "Irritability": 0,
    "delayed healing": 0,
    "partial paresis": 1,
    "muscle stiffness": 0,
    "Alopecia": 0,
    "Obesity": 0
}

# Serialize input data to JSON string
payload = json.dumps(input_data)

# Invoke the endpoint
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Accept='application/json',
    Body=payload
)

# Read the response
result = response['Body'].read().decode()
print("Response Content-Type:", response['ContentType'])
print("Prediction:", result)
