# Use the official Bitnami Spark image as the base image
FROM apache/spark:3.5.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt /app/ 

# Install dependencies with elevated permissions, then drop back to non-root user
USER root
RUN pip3 install --no-cache-dir -r requirements.txt
USER spark

# Copy the application code
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]