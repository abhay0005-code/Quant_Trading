FROM python:3.12-slim

WORKDIR /app

# Build + runtime deps for scipy, statsmodels, arch, lightgbm, xgboost.
#   - gcc/g++/gfortran: C/Fortran compilers for sdist builds (statsmodels/arch)
#   - cmake:               lightgbm builds from source via CMake when no wheel exists
#   - libopenblas-dev:     BLAS backend for scipy/numpy
#   - libgomp1:            OpenMP runtime lightgbm/xgboost link against at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ gfortran cmake \
        libopenblas-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Bound BLAS/OpenMP threads to 1 CPU to avoid thread oversubscription,
    # which multiplies per-thread stack allocations and OOMs small Railway
    # memory limits during import. Also speeds up startup.
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    XGBOOST_NUM_THREADS=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
