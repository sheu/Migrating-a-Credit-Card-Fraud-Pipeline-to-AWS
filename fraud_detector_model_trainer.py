from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, hour, when
from pyspark.sql.types import DoubleType
import boto3
import os


def MyTransform(glueContext, dfc):
    try: 
        # Convert DynamicFrameCollection to DataFrame
        selected_key = list(dfc.keys())[0]
        dynamic_frame = dfc[selected_key]
        df = dynamic_frame.toDF()

        # Convert all columns except 'Class' to DoubleType
        numeric_columns = [col for col in df.columns if col != 'Class']
        for column in numeric_columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))

        # Normalize numeric features
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=True)

        # Combine scaled numeric features
        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features")

        # Handle class imbalance by adjusting class weights
        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {row['Class']: total_count / row['count'] for row in class_counts}
        
        # Add class weights to the DataFrame
        df = df.withColumn("weight", when(col("Class") == 0, weight_dict[0]).otherwise(weight_dict[1]))

        # Convert Class to numeric and create class weights
        indexer = StringIndexer(inputCol="Class", outputCol="label")
        
        # Define the RandomForestClassifier with class weights
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, 
                                    maxDepth=10, weightCol="weight")

        # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])

        # Fit the model
        model = pipeline.fit(df)

        # Save model locally then upload to S3
        local_model_path = '/tmp/fraud_detection_model_latest'
        model.write().overwrite().save(local_model_path)

        print(f'Model trained and saved to {local_model_path}')

        # Upload local model to S3
        s3_bucket = os.environ.get('MODEL_S3_BUCKET', 'fraud-detection-model-bucket')
        s3_prefix = os.environ.get('MODEL_S3_PREFIX', 'model/fraud_detection_model_latest')
        s3 = boto3.client('s3')
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_model_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                s3.upload_file(local_file, s3_bucket, s3_key)
        print(f'Model uploaded to s3://{s3_bucket}/{s3_prefix}')

        # Convert DataFrame back to DynamicFrame and return DynamicFrameCollection
        result_dynamic_frame = DynamicFrame.fromDF(df, glueContext, "result")
        return DynamicFrameCollection({"results": result_dynamic_frame}, glueContext)
    except Exception as e:
        print(f"Error: {e}")
