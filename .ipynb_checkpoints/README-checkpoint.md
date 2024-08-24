# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pre-trained model that can perform image classification.This project is designed to simulate a typical machine learning workflow that a professional ML Engineer would undertake. The primary goal of this project is to demonstrate the setup of an ML infrastructure that can facilitate the training of accurate models by you or other developers.

## Project Set-up
Access AWS through WAS Gateway and open Sagemaker and create a folder for your project. Download the starter files from the project template provided n the classroom or you can clone the Github Repo.

## Dataset
Dog breed classification dataset  used was obtained from the classroom. The dataset is used to classify between different breeds of dogs in images. 

### Training and Hyperparameter Tuning
I firstly installed necessary dependencies  to read and preprocess data. I used the following hyperparameter ranges;
"learning_rate": ContinuousParameter(0.001, 0.1),
"batch_size": CategoricalParameter([32, 64, 128, 256]),

Below is a  list of the training jobs 

![Alt text](training_jobs.png?raw=true "training_jobs.png")
 

Hyperparameter tuning jobs completed as shown below,

Hyperparameter tuning jobs completed jobs:
![Alt text](hyperparameter_tuning.png?raw=true "hyperparameter_tuning.png")
 
Logs from the last completed training job with metrics during the process:

 ![Alt text](logs.png?raw=true "logs.png")

## Model Deployment
The model was deployed using ml.m5.xlarge EC2 instance and used Windows Server 2016 English Deep Learning for Amazon Machine Image (AMI) for training. The deployed endpoint is saved in the folder as 'Endpoint.png'

![Alt text](Endpoint.png?raw=true "Endpoint.png")
For querying the endpoint is needed to add the path of test data image by opening for reading this file and triggering predict function of model.

