# 1. Use a stable Python version on a slim Debian base
FROM python:3.12-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install updated system dependencies
# Changed libgl1-mesa-glx to libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libasound2 \
    libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy requirements first (Better for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application
COPY . .

# 7. Expose the port
EXPOSE 5000

# 8. Command to run your app (Check if your file is app.py or main.py)
CMD ["python3", "main.py"]