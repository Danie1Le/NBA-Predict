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
ENV PREBUILD_TRADITIONAL_ONLY=1

# Pre-build only traditional models during Docker build (memory efficient)
RUN python prebuild_models.py

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/ || exit 1

# Run the application with optimized startup
CMD ["python", "start_optimized.py"]
