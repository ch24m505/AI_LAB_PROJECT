# 1. Start from a Python 3.12 base image to match your training environment
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install Java (JDK), which is required by PySpark
RUN apt-get update && apt-get install -y default-jdk

# 4. Copy your new, clean requirements file
COPY requirements.txt .

# 5. Install the Python dependencies from the file
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code and the trained model artifacts
COPY ./src ./src
COPY ./models ./models

# 7. Expose the port the API will run on
EXPOSE 8000

# 8. Define the command to run your API when the container starts
CMD ["uvicorn", "src.fast_api_app:app", "--host", "0.0.0.0", "--port", "8000"]