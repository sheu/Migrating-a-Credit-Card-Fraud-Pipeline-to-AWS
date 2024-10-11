#TODO: Add the necessary Cloud SDK imports 
import os
import datetime
import time
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from fraud_detection_pipeline.fraud_detector_model_trainer import MyTransform

#TODO: Initiate the appopriate Cloud SDK Clients
spark = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()

#TODO: Modify the function to work with the apporpriate Cloud Storage location
def watch_directory_and_retrain(data_directory):
    # TODO: modify the function initiate based on the event in the cloud storage account
    while True:
        try:
            # Watch for new files in the specified directory
            files = os.listdir(data_directory)
            for file in files:
                if file.endswith('.csv') and 'retrain' not in file:
                    # TODO: modify the imports to call the service on the data in the cloud storage location
                    new_data_path = os.path.join(data_directory, file)
                    new_data = spark.read.csv(new_data_path, header=True, inferSchema=True)
                    print(f"New data detected: {new_data_path}")

                    # TODO: modify the function to kick off the AWS Service Job
                    MyTransform(new_data)
                    
                    # TODO: modify the function to move the processed file to an archive directory in the cloud storage location
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    retrain_file_path = os.path.join(data_directory, f"{os.path.splitext(file)[0]}_retrain_{current_date}.csv")
                    os.rename(new_data_path, retrain_file_path)
                    print(f"Renamed processed file to: {retrain_file_path}")

            
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(10)  # Check the directory every 10 seconds`

def main():
    # Define the data directory to watch
    data_directory = '/Users/jonathandyer/Documents/Dyer Innovation/data'
    # Call the function to watch the directory and retrain
    watch_directory_and_retrain(data_directory)

if __name__ == "__main__":
    main()

