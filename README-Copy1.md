<h3>Image Classification using AWS SageMaker.</h3>

Use AWS Sagemaker to train a pre-trained model that can perform image classification.This project is designed to simulate a typical machine learning workflow that a professional ML Engineer would undertake. The primary goal of this project is to demonstrate the setup of an ML infrastructure that can facilitate the training of accurate models by you or other developers.

<h3>Project Set-up</h3>

Access AWS through WAS Gateway and open Sagemaker and create a folder for your project. Download the starter files from the project template provided n the classroom or you can clone the Github Repo.

<h3>Dataset.</h3>

Dog breed classification dataset  used was obtained from the classroom. The dataset is used to classify between different breeds of dogs in images

<h3>Training</h3>

I firstly installed necessary dependencies  to read and preprocess data. I used the following hyperparameter ranges;
"learning_rate": ContinuousParameter(0.001, 0.1),
"batch_size": CategoricalParameter([32, 64, 128, 256]),

A list of the training jobs is saved in the folder as 'training_jobs.png'.

Hyperparameter tuning jobs completed are saved in the folder as 'hyperparameter_tuning.PNG'

Logs from the last completed training jobs with metrics obtained during the process are saved in the folder as 'logs.png'


<h3>Model Deployment</h3>

The model was deployed using ml.m5.xlarge EC2 instance and used Windows Server 2016 English Deep Learning for Amazon Machine Image (AMI) for training. The deployed endpoint is saved in the folder as 'Endpoint.png'


```python

```
