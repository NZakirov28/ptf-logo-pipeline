FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender1 git curl \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установка PyTorch с поддержкой CUDA
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Python зависимости
RUN pip install streamlit ultralytics \
    opencv-python-headless requests pyyaml \
    lama-cleaner

# Установка LabelMe
RUN npm install -g @labelme/web

# Копирование проекта
WORKDIR /app
COPY . /app

# Открытие портов
EXPOSE 8080 8081 8501

CMD npx labelme-web --host 0.0.0.0 --port 8080 & \
    python3 -m lama_cleaner.api --host 0.0.0.0 --port 8081 & \
    streamlit run app.py --server.port 8501 --server.headless true
