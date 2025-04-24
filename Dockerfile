FROM python:3.10-alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create a non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set up working directory
WORKDIR /app

# Install security updates and dependencies
RUN apk update && \
    apk upgrade && \
    apk add --no-cache gcc musl-dev linux-headers && \
    apk add --no-cache libffi-dev

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -e .

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
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
