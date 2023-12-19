# Use the official Python 3.10 slim image as the base
FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set the working directory to /app
WORKDIR /app

# Copy only the dependency definition files to the working directory
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install project dependencies using Poetry
RUN poetry install --no-root --no-dev

# Copy the application files to the working directory
COPY ["predict.py", "model.bin", "./"]

# Expose port 9696
EXPOSE 9696

# Set the entry point to start the application using Waitress
ENTRYPOINT ["poetry", "run", "waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]