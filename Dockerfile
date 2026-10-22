# Class Quality Assessor: Docker image for a Hugging Face Space (Streamlit on port 7860).
# Nothing confidential is copied in (.dockerignore); the DeepSeek key comes from a Space secret at runtime.

# --- Stage 1: build whisper.cpp ---
FROM python:3.11-slim AS whisper
RUN apt-get update && apt-get install -y --no-install-recommends cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# v1.9.5 is the same commit as tag b5454: the build used for the local benchmarks (EXPERIMENTS E8, DECISIONS D11)
ARG WHISPER_TAG=v1.9.5
ARG WHISPER_COMMIT=d1be6fde11ac6e0407606b4e42fe72d34add8037
RUN git clone --depth 1 --branch ${WHISPER_TAG} https://github.com/ggml-org/whisper.cpp /src/whisper.cpp \
    && test "$(git -C /src/whisper.cpp rev-parse HEAD)" = "${WHISPER_COMMIT}"
# Portable x86-64 build (AVX2/FMA, no -march=native): the build machine is not the machine the Space runs on
RUN cmake -S /src/whisper.cpp -B /src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
    && cmake --build /src/build --config Release -j"$(nproc)" --target whisper-cli \
    && cp /src/build/bin/whisper-cli /usr/local/bin/whisper-cli

# --- Stage 2: app ---
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=whisper /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli

# Models, checked against the files used in development
RUN mkdir -p /opt/models \
    && curl -fsSL -o /opt/models/face_detection_yunet_2023mar.onnx \
        https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    && curl -fsSL -o /opt/models/ggml-base-q5_1.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin \
    && printf '%s  %s\n' \
        8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 /opt/models/face_detection_yunet_2023mar.onnx \
        422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898 /opt/models/ggml-base-q5_1.bin \
        | sha256sum -c -

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    FFMPEG_BIN=ffmpeg \
    FFPROBE_BIN=ffprobe \
    WHISPER_CLI=/usr/local/bin/whisper-cli \
    WHISPER_MODEL=/opt/models/ggml-base-q5_1.bin \
    WHISPER_THREADS=2 \
    FACE_MODEL=/opt/models/face_detection_yunet_2023mar.onnx \
    CACHE_DIR=/home/user/app/.cache \
    DELETE_MEDIA_AFTER_RUN=true \
    CACHE_MAX_AGE_H=2
WORKDIR /home/user/app
COPY --chown=user . .

EXPOSE 7860
# XSRF and CORS off: Streamlit uploads otherwise fail with 403 inside the Hugging Face iframe
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]
