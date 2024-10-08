# How are you feeling?
#### Cloud Engineering Final Project

Contributors: Gabriel Zhang, Kate Yee, Xenia Vrettakou & Zoe Li

## Purpose
Addressing mental health concerns has become increasingly important in the persisting isolation of the post-pandemic period. Our goal for this project was to create a cloud based web app to enable early detection of emotional distress to proactively provide students with counselling services.


## Architecture
![Architecture](./images/architecture.png)

Our architecture can be easily split into three processes:

**1. Training Data Augmentation** - When an image training data set is uploaded to the S3 Raw bucket, a lambda function is triggered to ingest that dataset, create an augmented image dataset, and upload the resulting augmented dataset to the S3 Refined bucket.

**2. Model Training** - The model training pipeline is containerized in ECR and can be deployed as an ECS task to ingest an augmented dataset from the S3 Refined bucket, train, evaluate, and score a model. Then upload a pickled model and related artifacts to the S3 Model Storage bucket.

**3. Inference Web Application** - The web application is deployed as a streamlit app containerized in an ECS service. A user can upload a raw image to the web app that the service will send to a lambda function for preprocessing. The preprocessed image is then returned to the ECS service, which pulls a model from the S3 Model Storage bucket, predicts an emotion from the given image, and returns the inferred emotion to be displayed on the web app to the user.

[Link to our cost estimation for this architecture.](https://calculator.aws/#/estimate?id=b1b1ae6e15b5c739a3bd9e2c219bd8ac910979f6)

### Data
To train our model, we found a [Kaggle dataset for facial emotional recognition](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/overview).


## Repository Overview
This repository contains all the code necessary to deploy on AWS the architecture previously described. The contents of the repository are detailed below:

- `app/`: The directory containing the script and resources for running the streamlit web application.
- `dockerfiles/`: The directory containg dockerfiles for building the web app.
- `images/`: The directory containg images referenced in this README.
- `pipeline/`: The directory containing the model training pipeline scripts and associated resources:
  - `config/`: The directory containing the model training pipeline configuration files.
  - `logs/`: The directory containing the model training pipeline log files.
  - `src/`: The directory containing the model training pipeline python module scripts and the `main.py` script.
  - `tests/`: The directory containing the model training pipeline unit tests.
  - `Dockerfile`: The Dockerfile for building the model training pipeline imgae.
  - `requirements.txt`: The model training pipeline package requirements.
- `preprocessing_lambda/`: The directory containing the script used to augment training data.
- `preprocessing_lambda_inference/`: The directory containing the script used to augment user uploaded data for inference.
- `.gitignore`: The file detailing untracked files Git should ignore.
- `.pylintrc`: The file containing the lintr standard configurations for this repository.
- `README.md`: The README you're reading right now.

### Pipeline Unittests
To run the pipeline unittests, navigate to the pipeline folder and run the following line of code:
```
python -m unittest discover -s tests -p '*test*.py'
```
To receive this result:

![Unit Tests](./images/pipeline_unittest.png)

## Deployment Overview
The steps for deploying 

### Training Data Augmentation
![Phase1](./images/phase1.png)

1. Create a lambda function with S3 permissions for the execution role, the `preprocessing_lambda/main.py` script, a trigger on created objects in the S3 Raw bucket, 3004 MB memory, and 2048 MB ephemeral storage.

![Augmentation Lambda Function](./images/augmenting_lambda_config.png)

2. Add three layers to the function to enable the pandas, numpy, and pillow libraries using [these ARN's](https://api.klayers.cloud/api/v2/p3.12/layers/latest/us-east-2/html).

3. Upload a training data csv to the S3 Raw bucket and wait for the augmented data csv to appear in the S3 Refined bucket.

![Augmented Dataset](./images/augmented_data.png)

### Model Training
![Phase2](./images/phase2.png)

1. Create a new ECR registry and use the push commands from the `pipeline/` directory to build and push the model training pipeline to ECR.
   
2. Create a new ECS cluster to run tasks.
  
3. Create a new task definiion to run the model training pipeline image in ECR using Faregate.

4. Deploy the task in the ECS cluster and wait for the pipeline to deploy and run.

![Model Training Pipeline Completion](./images/model_pipeline_completion.png)

During that process, the trained model and associated artifacts will be uploaded to the S3 Model Storage bucket.

![Model Training Artifacts](./images/model_artifacts.png)

### Inference Web Application
![Phase3](./images/phase3.png)


1. Create AWS Lambda function to preprocess user uploaded image for model inference.

2. Make the web app to load model from the S3 bucket dynamically, and invoke Lambda function using boto3 in Python when received a new image.

2. Build and push docker image of the web app in the `app` directory to ECR.

3. Create a new ECS cluster for app hosting and model inference.

4. Define task and deploy it as service with necessary permissions, port mappings, and networking security group.

6. Access the app from the public IP, and upload facial images to get emotion prediction.
