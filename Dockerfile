# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Upgrade pip and install the application dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip list

# Copy the rest of the application source code to the working directory
COPY . .

# Expose the port that the app will run on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "Dashb_parassit.py", "--server.port=8501"]
