# =============================================================================
# SkinAge — Multi-stage Docker build for production inference
# =============================================================================
# Target image size: < 4 GB
# Usage:
#   docker build -t skinage:latest .
#   docker run -p 8000:8000 -v ./outputs/models:/app/outputs/models skinage:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System dependencies required for building wheels (opencv, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (smaller than CUDA bundle)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source and configuration
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Model weights are mounted as a volume at runtime
VOLUME ["/app/outputs/models"]

# FastAPI default port
EXPOSE 8000

# Health check: hit the API health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
