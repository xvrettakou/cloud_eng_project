## Run Dockerfile for web app 
```
docker build -t fer-app -f dockerfiles/Dockerfile-app .
docker run -p 80:80 fer-app
```