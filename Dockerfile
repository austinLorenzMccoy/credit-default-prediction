FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create a non-root user
RUN groupadd --system appgroup && useradd --system --create-home --gid appgroup appuser

# Set up working directory
WORKDIR /app

# Install security updates and dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y gcc build-essential linux-headers-generic && \
    apt-get install -y libffi-dev && \
    apt-get clean

# Copy requirements and install dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p models && \
    chown -R appuser:appgroup /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Expose the port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Run the application with reduced privileges
CMD ["uvicorn", "backend.app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
