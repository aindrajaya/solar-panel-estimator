# Use the official Python 3.6 image
FROM python:3.6-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ .

# Expose the necessary port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
