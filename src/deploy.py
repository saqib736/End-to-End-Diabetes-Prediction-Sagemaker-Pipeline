import boto3
import time

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')

    endpoint_name = event['EndpointName']
    model_package_arn = event['ModelPackageArn']
    execution_role_arn = event['ExecutionRoleArn']

    # Generate unique names for the model and endpoint configuration
    timestamp = int(time.time())
    model_name = f"{endpoint_name}-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    # Create Model
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=execution_role_arn
    )

    # Create Endpoint Configuration
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large'
            }
        ]
    )

    # List existing endpoints to check if the endpoint already exists
    existing_endpoints = sagemaker_client.list_endpoints(
        NameContains=endpoint_name
    )['Endpoints']

    endpoint_exists = any(
        ep['EndpointName'] == endpoint_name for ep in existing_endpoints
    )

    if endpoint_exists:
        # Update the existing endpoint with the new configuration
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Updated existing endpoint: {endpoint_name}")
    else:
        # Create a new endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Created new endpoint: {endpoint_name}")
