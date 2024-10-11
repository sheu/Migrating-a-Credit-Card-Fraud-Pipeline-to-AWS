# TODO: Import necessary AWS Glue libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, Bucketizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, hour, when
from pyspark.sql.types import DoubleType

# TODO: Remove this SparkSession creation for Glue
spark = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()

# TODO: Modify function signature for Glue
def MyTransform(df):
    try: 
        # TODO: Convert DynamicFrameCollection to DataFrame for Glue

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

        # TODO: Modify model saving for Glue (use S3 instead of local path)
        model_path = '../fraud_detection_api/model/fraud_detection_model_latest'
        model.write().overwrite().save(model_path)

        print(f'Model trained and saved to {model_path}')

        # TODO: Add S3 upload code here for Glue

        # TODO: Convert DataFrame back to DynamicFrame and return DynamicFrameCollection for Glue

        return df  # TODO: Modify return value for Glue
    except Exception as e:
        print(f"Error: {e}")

# TODO: Remove this main block for Glue
if __name__ == "__main__":
    # Load your CSV file
    df = spark.read.csv("../data/credit_card_transaction_data_labeled.csv", header=True, inferSchema=True)
    
    # Run the transformation
    result_df = MyTransform(df)

    # Stop the SparkSession
    spark.stop()