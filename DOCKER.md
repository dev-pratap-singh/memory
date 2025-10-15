# Docker Setup Guide

This project supports two Docker configurations:

## 1. macOS / Development (Default)

**File**: `Dockerfile`
**Platform**: macOS (Apple Silicon), Linux (AMD64)
**GPU**: CPU-only (no CUDA)

### Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Features
- ✅ Works on macOS (Apple Silicon M1/M2/M3)
- ✅ Works on Linux AMD64
- ✅ CPU-only PyTorch (lightweight)
- ✅ No GPU dependencies
- ✅ Smaller image size
- ✅ Faster builds

---

## 2. Linux Production with NVIDIA GPU

**File**: `Dockerfile.cuda`
**Platform**: Linux with NVIDIA GPU
**GPU**: CUDA 12.1 + cuDNN 8

### Quick Start

```bash
# Use the CUDA Dockerfile
export DOCKERFILE=Dockerfile.cuda
export USE_GPU=true

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app
```

### Features
- ✅ Full GPU acceleration
- ✅ CUDA 12.1 + cuDNN 8
- ✅ Optimized for training
- ✅ bitsandbytes quantization
- ❌ Only works on Linux with NVIDIA GPU
- ❌ Larger image size (~10GB)

### Requirements
- NVIDIA GPU
- NVIDIA Docker runtime installed
- Linux host

---

## Environment Variables

### Switching Between Dockerfiles

Create a `.env` file:

```bash
# For macOS / CPU-only (default)
DOCKERFILE=Dockerfile
USE_GPU=false

# For Linux with GPU
# DOCKERFILE=Dockerfile.cuda
# USE_GPU=true
```

### Database Configuration

```bash
DB_NAME=memory_db
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_PORT=5432
```

### API Keys

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI application |
| Metrics | 9090 | Prometheus metrics |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache & task queue |

---

## Common Commands

### Build and Start

```bash
# Rebuild and start
docker-compose up -d --build

# Start specific service
docker-compose up -d app

# Scale workers
docker-compose up -d --scale worker=3
```

### Debugging

```bash
# View logs
docker-compose logs -f app
docker-compose logs -f worker
docker-compose logs -f postgres

# Enter container
docker exec -it memory_app bash

# Check database
docker exec -it memory_postgres psql -U postgres -d memory_db

# Test Redis
docker exec -it memory_redis redis-cli ping
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

---

## Troubleshooting

### macOS: Build fails with GPU errors

**Solution**: Make sure you're using the default `Dockerfile` (not `Dockerfile.cuda`)

```bash
unset DOCKERFILE
docker-compose up -d --build
```

### Linux: Want to use GPU

**Solution**: Switch to `Dockerfile.cuda`

```bash
export DOCKERFILE=Dockerfile.cuda
export USE_GPU=true
docker-compose up -d --build
```

### Out of memory during build

**Solution**: Increase Docker memory limit

- Docker Desktop → Settings → Resources → Memory: 8GB+

### Port already in use

**Solution**: Change ports in `.env`

```bash
API_PORT=8001
DB_PORT=5433
REDIS_PORT=6380
```

---

## Performance Tips

### macOS / Development
- Use default `Dockerfile` for faster builds
- Set `BUILD_TARGET=development` for hot reload
- Limit worker concurrency: `WORKER_CONCURRENCY=2`

### Linux / Production
- Use `Dockerfile.cuda` for GPU acceleration
- Set `BUILD_TARGET=production` for optimized builds
- Increase workers: `WORKER_CONCURRENCY=4`

---

## Architecture

```
┌─────────────────────────────────────────┐
│  app (FastAPI)                          │
│  - Handles API requests                 │
│  - Uses GPU if available                │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼───┐ ┌───▼────┐
│worker │ │worker│ │  beat  │
│(celery│ │(celery│ │(celery)│
└───┬───┘ └──┬───┘ └───┬────┘
    │        │         │
    └────────┼─────────┘
             │
    ┌────────▼────────┐
    │ Redis           │
    │ (Queue/Cache)   │
    └─────────────────┘

    ┌─────────────────┐
    │ PostgreSQL      │
    │ + pgvector      │
    └─────────────────┘
```

---

## Next Steps

1. ✅ Start with `Dockerfile` (macOS-friendly)
2. ✅ Test locally: `docker-compose up -d`
3. ✅ Check health: `curl http://localhost:8000/health`
4. ✅ Deploy to Linux with GPU using `Dockerfile.cuda`
