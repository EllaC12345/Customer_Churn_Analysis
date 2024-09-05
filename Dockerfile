FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt requirements.txt


# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .


EXPOSE 8080

ENV PORT=8080
# Run gunicorn server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
