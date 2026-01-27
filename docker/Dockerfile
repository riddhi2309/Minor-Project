# 1. Base image
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy dependency file
COPY docker/requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY src/ src/

# 6. Default command
CMD ["python", "src/main.py"]
