# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container at /app
COPY godfathersprogram.py /app/

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the script
CMD ["python", "./godfathersprogram.py"]
