# Migrating a Credit Card Fraud Pipeline to AWS

In this project, you will be working as a cloud architect at a credit card company. The company's data scientists have developed a fraud detection pipeline on-premises, which includes:
1. A Spark pipeline running on a Hadoop cluster that transforms incoming data and trains the model.
2. A containerized API running on an on-premises Apache server that returns a fraud/no-fraud response.
3. A function that updates the data used to train the model with data every couple of months and kicks off retraining of the model.
You have been tasked with migrating this pipeline to AWS Cloud while ensuring improved security, scalability, performance, and cost-efficiency.. In order to do so you will use what you've learned in the course to:
1. Evaluate each application and determine which AWS Service will be best for deploying the service based on AWS best practices
2. Make the appropriate changes to the source code to prepare the applications for deployment to the appropriate AWS service.
3. Deploy and configure each application appropriately on AWS
4. Verify the application is functioning adequately AWS

## Getting Started

Follow the steps below to make modifications to the code on your local machine:
1. Clone the project repo 
2. Make the necessary modifications to the code where "#TODO"s have been provided
3. Upload the appropriate code to your AWS environment
4. Test your solution utilizing each AWS service

### Dependencies

```
Python 3.9
PySpark 3.3.1
FastAPI 0.89.1
Joblib 1.2.0
Pipenv 2022.10.4
```

### Installation

Step-by-step explanation of how to get a dev environment running.

List out the steps
1. Clone the repository to your local machine
2. (Optional) Install and create a virtual environment using pipenv
  - ```pip install pipenv```
  - ```pipenv shell```
2. Install the necessary dependencies using pipenv in each project directory
  - ```pipenv install pyspark fastapi joblib ```
  - Note: the directories are meant to be standalone pipenv environments, so you can run ```pipenv shell``` in each directory to activate the environment.
3. You can now run the code locally using the following command:
  - ```pipenv run python <app_name>.py```

## Testing

Each test script is provided within the tests folder. The tests are functions that are named verify_<service_name>_<purpose>.py. For example, the test for the model training job is named verify_glue_job.py. **TODO: Add a better name for each test that is more discreet.**

### Break Down Tests

Each test script is explained below:

1. verify_glue_job.py
  - This test verifies that the Glue job is configured correctly 
  - It also uses a test dataset to verify that the data is being written to the correct location.
2. verify_ecs_api.py
  - This test verifies that the ECS API is configured correctly and that the necessary permissions are granted.
  - It also uses a test request to verify that the API is working correctly.
3. verify_lambda_function.py
  - This test verifies that the Lambda function is configured correctly 
  - It also uses a test request to verify that the function is working correctly.
4. verify_iam_roles.py
  - This test verifies that the IAM roles are correctly configured and that the necessary permissions are granted.

To run the tests use:

```
pipenv run python tests/verify_<service_name>_<purpose>.py.
```

For example, to run the test for the model training job use:
```
pipenv run python tests/verify_glue_job.py
```
## Project Instructions

Below, each application is provided with its description and requirements. For each application, you must complete the following three tasks:
1. Identify the appropriate AWS service and configuration to migrate the application to based on the course. Write out your rationale in a doc titled service_rationale.docx.
2. Modify each provided Python script provided to work in that service. (TODOs have been provided for each script) and use the provided test scripts to verify your function has been created appropriately. 
3. Configure each AWS Service including provisioning the service and appropriate IAM Roles. 
4. Use the associated test scripts to verify the AWS Service is running appropriately. 
5. Once completed for each service, use the final test script to verify the application works correctly together. 

### Model Training Job
The development team's model training job utilizes PySpark to:
  - Extract the raw data from the source location 
  - Transform the data (splitting into train/test, modifying columns for training) 
  - Load the resulting data into the input folder 
  - Train the model and output the resulting model in the destination folder
The development team would like an AWS Service that satisfies the following requirements:
  - Automate as much as possible of the data preparation environment. 
  - Integrate with other AWS Services. 
  - The team does not have experience managing infrastructure so they would like to use a fully managed service. 
  - The model will only be trained monthly, so they don't want to pay for ongoing infrastructure. 
  - Eventually, they would like to develop a data catalog of all of their existing datasets. 

### Model Deployment API
Once the model is trained, the development team utilizes the trained model via a fast API service to:
  - read the incoming transaction
  - process the transaction as fraud or non-fraud 
  - return the reading at the endpoint 
The development team would like to deploy their API to an AWS service that:
  - Can serve as a container orchestration service without the need to manage servers or Kubernetes clusters. 
  - Can automatically scale to support an increased load of requests.
  - Integrates with CloudWatch and CloudTrail for monitoring and logging. 

### Model Retraining Function
The model retraining function is an event-based function that:
  - watches a local directory for a new set of data to retrain the model.
  - when a new file is placed in the directory it kicks off re-training of the model.
  - The model is then labeled as latest and saved in the appropriate directory to be used for inference.
They would like an AWS service that:
  - Does not require the developers to manage servers.
  - Can monitor a database for changes and run the function when data is loaded. 
  -  Is efficient for this infrequent monthly workload
Can easily integrate with the service used for their job.

## Built With

* [AWS Simple Storage Service](https://aws.amazon.com/s3/) - Object storage for data
* [AWS Glue](https://aws.amazon.com/glue/) - Data Catalog
* [AWS Lambda](https://aws.amazon.com/lambda/) - Serverless compute
* [AWS Elastic Container Registry](https://aws.amazon.com/ecr/) - Container registry
* [AWS Elastic Container Service](https://aws.amazon.com/ecs/) - Container orchestration
* [AWS CloudWatch](https://aws.amazon.com/cloudwatch/) - Monitoring and logging
* [AWS CloudTrail](https://aws.amazon.com/cloudtrail/) - Monitoring and logging
* [AWS Cloud9](https://aws.amazon.com/cloud9/) - Cloud-based IDE

Include all items used to build the project.

## License
[License](../LICENSE.md)
