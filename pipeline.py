# pipeline.py

import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterFloat
)
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.lambda_helper import Lambda
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.functions import Join
from sagemaker.sklearn.model import SKLearnModel 

def get_pipeline(role):
    sagemaker_session = PipelineSession()
    
    output_bucket = "s3://diabetes-prediction-pipeline"  

    # Parameters
    input_data = ParameterString(name="InputData", default_value="s3://diabetes-prediction-pipeline/diabetes_data.csv")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.8)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_package_group_name = ParameterString(name="ModelPackageGroupName", default_value="diabetes-prediction-models")
    endpoint_name = ParameterString(name="EndpointName", default_value="DiabetesPredictionEndpoint")

    # Preprocessing Step
    sklearn_processor = ScriptProcessor(
        role=role,
        image_uri=sagemaker.image_uris.retrieve(
            framework='sklearn',
            region=sagemaker_session.boto_region_name,
            version='0.23-1'
        ),
        command=['python3'],
        instance_type='ml.m5.large',
        instance_count=1,
        sagemaker_session=sagemaker_session
    )

    preprocessing_output_destination = f"{output_bucket}/preprocessing"
    preprocessing_step = ProcessingStep(
        name="PreprocessingStep",
        processor=sklearn_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output/train",
                destination=f"{preprocessing_output_destination}/train"
            ),
            sagemaker.processing.ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/output/test",
                destination=f"{preprocessing_output_destination}/test"
            )
        ],
        code="src/preprocessing.py",
        job_arguments=[
            "--input_path", "/opt/ml/processing/input/diabetes_data.csv",
            "--train_output_path", "/opt/ml/processing/output/train/train_data.csv",
            "--test_output_path", "/opt/ml/processing/output/test/test_data.csv"
        ]
    )

    # Training Step
    training_output_path = f"{output_bucket}/training"
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_type='ml.m5.large',
        framework_version='0.23-1',
        py_version='py3',
        sagemaker_session=sagemaker_session,
        output_path=training_output_path
    )
    
    training_step = TrainingStep(
        name="TrainingStep",
        estimator=sklearn_estimator,
        inputs={
            'train': sagemaker.inputs.TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    # Evaluation Step
    evaluation_processor = ScriptProcessor(
        role=role,
        image_uri=sklearn_processor.image_uri,
        command=['python3'],
        instance_type='ml.m5.large',
        instance_count=1,
        sagemaker_session=sagemaker_session
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    evaluation_output_destination = f"{output_bucket}/evaluation"
    evaluation_step = ProcessingStep(
        name="EvaluationStep",
        processor=evaluation_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            sagemaker.processing.ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"{evaluation_output_destination}/evaluation_report"
            )
        ],
        code="src/evaluate.py",
        property_files=[evaluation_report],
        job_arguments=[
            "--model_path", "/opt/ml/processing/model/model.tar.gz",
            "--test_data_path", "/opt/ml/processing/test/test_data.csv",
            "--report_path", "/opt/ml/processing/evaluation/evaluation.json"
        ]
    )

    # Condition Step
    cond_gte_accuracy = sagemaker.workflow.conditions.ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy"
        ),
        right=accuracy_threshold
    )

    # Model Registration Step
    model = SKLearnModel(
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point='inference.py',
        source_dir='src',
        framework_version='0.23-1',
        sagemaker_session=sagemaker_session
    )

    model_metrics = sagemaker.model_metrics.ModelMetrics(
        model_statistics=sagemaker.model_metrics.MetricsSource(
            s3_uri=Join(
                        on="/",
                        values=[
                            evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                            "evaluation.json"
                        ]
                    ),
            content_type="application/json"
        )
    )

    register_step = RegisterModel(
        name="RegisterModelStep",
        model=model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )

    # Deployment Step using Lambda
    lambda_role = 'arn:aws:iam::account_id:role/DiabetesSageMakerLambdaExecutionRole'

    lambda_function = Lambda(
        function_name="sagemaker-deploy-model",
        execution_role_arn=lambda_role,
        script="src/deploy.py",
        handler="deploy.lambda_handler",
        timeout=600,
        memory_size=128
    )

    deployment_step = LambdaStep(
        name="DeploymentStep",
        lambda_func=lambda_function,
        inputs={
            "ModelPackageArn": register_step.properties.ModelPackageArn,
            "EndpointName": endpoint_name,
            "ExecutionRoleArn": role
        }
    )

    # Conditional Step
    condition_step = ConditionStep(
        name="AccuracyConditionStep",
        conditions=[cond_gte_accuracy],
        if_steps=[register_step, deployment_step],
        else_steps=[]
    )

    # Pipeline Definition
    pipeline = Pipeline(
        name="DiabetesPredictionPipeline",
        parameters=[
            input_data,
            accuracy_threshold,
            model_approval_status,
            model_package_group_name,
            endpoint_name
        ],
        steps=[preprocessing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=sagemaker_session
    )

    return pipeline

if __name__ == "__main__":
    role = 'arn:aws:iam::account_id:role/DiabetesSageMakerExecutionRole'
    pipeline = get_pipeline(role)
    pipeline.upsert(role_arn=role)
