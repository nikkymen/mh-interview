# Build dependencies image
FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    cmake \
    ninja-build \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libdlib-dev \
    libgtk2.0-dev \
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy source code
COPY ./openface /workspace
WORKDIR /workspace

# Build OpenFace
RUN cmake -B ./build/ -S ./ -DCMAKE_INSTALL_PREFIX=/usr/local -G Ninja && \
    cmake --build ./build/ --target install

FROM ubuntu:24.04

ENV DOCKER_IMAGE_VERSION=0.0.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    sox \
    libopencv-core406t64 \
    libopencv-imgproc406t64 \
    libopencv-calib3d406t64 \
    libopencv-highgui406t64 \
    libopencv-objdetect406t64 \
    libopencv-videoio406t64 \
    libdlib19.1t64 \
    liblapack3 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Copy application code

COPY uv.lock .
COPY pyproject.toml .

RUN uv sync --locked

COPY pipeline.py .

# Copy built binaries from builder stage
COPY --from=builder /usr/local/bin /usr/local/bin

ENTRYPOINT ["uv", "run", "pipeline.py"]