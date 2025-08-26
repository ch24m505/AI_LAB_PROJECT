# 1. Start from a Python 3.12 base image to match your training environment
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install the default Java (JDK), which is required by PySpark
RUN apt-get update && apt-get install -y default-jdk

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code into the container
COPY ./src ./src

# 7. Expose the port the API will run on
EXPOSE 8000

# 8. Define the command to run your API when the container starts
CMD ["uvicorn", "src.fast_api_app:app", "--host", "0.0.0.0", "--port", "8000"]