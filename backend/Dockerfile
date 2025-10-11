# Optimized single-stage build for memory efficiency
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend files
COPY *.py .
COPY Data/ ./Data/

# Create model_cache directory (will be populated during prebuild)
RUN mkdir -p model_cache

# Set memory limits for model training
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose port (Render will set PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Run the application
CMD ["python", "main.py"]
