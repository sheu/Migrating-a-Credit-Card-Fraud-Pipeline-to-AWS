import json
import sys
from datetime import datetime, timezone

import boto3
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith('s3://'):
        raise ValueError('MODEL_SAVE_PATH must start with s3://')

    path = s3_uri[5:]
    bucket, separator, key = path.partition('/')
    if not bucket or not separator or not key:
        raise ValueError('MODEL_SAVE_PATH must be in the format s3://bucket/prefix')

    return bucket, key.rstrip('/')


def write_latest_marker(model_save_path: str) -> None:
    bucket, model_prefix = parse_s3_uri(model_save_path)
    marker_key = f'{model_prefix}/latest_marker.json'
    marker_payload = {
        'model_path': model_save_path,
        'updated_at_utc': datetime.now(timezone.utc).isoformat()
    }

    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket,
        Key=marker_key,
        Body=json.dumps(marker_payload).encode('utf-8'),
        ContentType='application/json'
    )


def MyTransform(glueContext, dfc, model_save_path):
    try:
        input_dynamic_frame = list(dfc.values())[0]
        df = input_dynamic_frame.toDF()

    # Convert all columns except 'Class' to DoubleType
        numeric_columns = [col for col in df.columns if col != 'Class']
        for column_name in numeric_columns:
            df = df.withColumn(column_name, col(column_name).cast(DoubleType()))

    # Normalize numeric features
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=True)

    # Combine scaled numeric features
        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features")

    # Handle class imbalance by adjusting class weights
        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {row['Class']: total_count / row['count'] for row in class_counts}

        class_labels = list(weight_dict.keys())
        if len(class_labels) != 2:
            raise ValueError('Expected exactly two classes in column Class for binary classification.')
    
    # Add class weights to the DataFrame
        df = df.withColumn(
            "weight",
            when(col("Class") == class_labels[0], weight_dict[class_labels[0]]).otherwise(weight_dict[class_labels[1]])
        )

    # Convert Class to numeric and create class weights
        indexer = StringIndexer(inputCol="Class", outputCol="label")
    
    # Define the RandomForestClassifier with class weights
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100,
                                    maxDepth=10, weightCol="weight")

    # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])

    # Fit the model
        model = pipeline.fit(df)

        model.write().overwrite().save(model_save_path)

        print(f'Model trained and saved to {model_save_path}')

        write_latest_marker(model_save_path)
        print(f'Model marker updated in {model_save_path}')

        transformed_dynamic_frame = DynamicFrame.fromDF(df, glueContext, 'transformed_df')

        return DynamicFrameCollection({'transformed_df': transformed_dynamic_frame}, glueContext)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'DATA_URL', 'MODEL_SAVE_PATH'])

    spark_context = SparkContext.getOrCreate()
    glue_context = GlueContext(spark_context)
    job = Job(glue_context)
    job.init(args['JOB_NAME'], args)

    input_dynamic_frame = glue_context.create_dynamic_frame.from_options(
        connection_type='s3',
        connection_options={'paths': [args['DATA_URL']]},
        format='csv',
        format_options={'withHeader': True, 'inferSchema': True}
    )

    input_dynamic_frame_collection = DynamicFrameCollection({'input_data': input_dynamic_frame}, glue_context)
    MyTransform(glue_context, input_dynamic_frame_collection, args['MODEL_SAVE_PATH'])

    job.commit()