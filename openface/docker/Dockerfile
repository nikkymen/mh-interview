# Build stage
FROM nikkymen/tarsius-dev-base:latest AS builder

USER root

ARG DEBIAN_FRONTEND=noninteractive

# Install APT packages needed for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    cmake \
    ninja-build \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install opencv
ADD docker/libs.tar.gz /usr/local

# Copy source code
COPY . /workspace
WORKDIR /workspace

# Build OpenFace
RUN cmake -B /build/release -S ./ -DCMAKE_INSTALL_PREFIX=/usr/local -G Ninja && \
    cmake --build /build/release --target install

# Runtime stage
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgtk2.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libtbb2 \
    libjpeg8 \
    libpng16-16 \
    libtiff5 \
    libdc1394-22 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy built binaries from builder stage
COPY --from=builder /usr/local/bin /usr/local/bin

# Install OpenFace models
ADD docker/models.tar.gz /usr/local/etc/OpenFace

WORKDIR /data

CMD ["/bin/bash"]