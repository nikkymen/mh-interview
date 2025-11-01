FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends\
    ffmpeg \
    curl \
    ca-certificates \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Copy application code

COPY uv.lock .
COPY pyproject.toml .
COPY get_features.py .
COPY audio_features/extract_audio.py ./audio_features/
COPY audio_features/opensmile_features.py ./audio_features/
COPY video_features/openface_features.py ./video_features/
COPY video_features/tsfresh_features.py ./video_features/
COPY text_features/llm_features.py ./text_features/
COPY transcript/whisper_transcript.py ./transcript/

RUN uv sync --locked

#TODO add models

# Set entrypoint
ENTRYPOINT ["uv", "run", "get_features.py"]