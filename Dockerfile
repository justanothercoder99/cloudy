# Use the official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# environment variables for no __pycache__ and direct log to stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

# Copy project files into the container
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev
RUN pip install -r requirements.txt

# Run the web server
CMD ["gunicorn", "--limit-request-line", "8190", "--limit-request-field_size", "8190", "-b", "0.0.0.0:8080", "-w", "1", "-t", "120", "app:app"]