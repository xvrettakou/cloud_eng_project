# How are you feeling?
Cloud Engineering Final Project
Contributors: Gabriel Zhang, Kate Yee, Xenia Vrettakou & Zoe Li

## Architecture

![Architecture](./images/CloudEngineering.png)

Cost Estimation:

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
