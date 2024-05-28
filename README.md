# How are you feeling?
#### Cloud Engineering Final Project

Contributors: Gabriel Zhang, Kate Yee, Xenia Vrettakou & Zoe Li

## Purpose
Addressing mental health concerns has become increasingly important in the persisting isolation of the post-pandemic period. Our goal for this project was to create a cloud based web app to enable early detection of emotional distress to proactively provide students with counselling services.


## Architecture
![Architecture](./images/CloudEngineering.png)
Our architecture 


Cost Estimation: ***link***

## Repos Overview

## Deployment Overview

### Preprocessing Lambda

### Model Training

### Web Application

#### Run Dockerfile for web app 
```
docker build -t fer-app -f dockerfiles/Dockerfile-app .
docker run -p 80:80 fer-app
```
