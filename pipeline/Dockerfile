# Use an official Python runtime as a parent image
FROM --platform=linux/x86_64 python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Connect to source code and configuration file
COPY config config 
COPY src src

# Run main.py when the container launches
CMD ["python", "src/main.py"]
