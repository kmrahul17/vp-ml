FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .  # Copy requirements file first to leverage Docker cache
RUN pip install -r requirements.txt  # Install dependencies

COPY . .  # Copy the rest of your application files into the container
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]  # Start Flask app with gunicorn
