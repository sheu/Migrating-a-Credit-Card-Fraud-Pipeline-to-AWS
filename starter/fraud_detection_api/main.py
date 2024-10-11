from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors


class InputData(BaseModel):
    features: list[float]

app = FastAPI()

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionAPI").getOrCreate()

model = None

@app.on_event('startup')
async def load_model():
    global model
    #TODO: Update to use the appopriate Cloud Storage location
    model_path = 'model/fraud_detection_model_latest' 
    #TODO: Add an intermediate storage location

    # Load the PySpark model
    model = PipelineModel.load(model_path)

@app.post('/predict/')
def predict(data: InputData):
    # Declaring Feature Columns
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    # Create a Spark DataFrame for prediction
    df = spark.createDataFrame([data.features], columns)

    # Make prediction
    predictions = model.transform(df)

    prediction = predictions.select("prediction").collect()[0][0]
    
    return {'prediction': prediction}
