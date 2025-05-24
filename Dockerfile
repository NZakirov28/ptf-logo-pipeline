FROM python:3.10-slim

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx npm \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Устанавливаем LabelMe Web через npm
RUN npm install -g @labelme/web

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir \
    streamlit \
    ultralytics \
    opencv-python-headless \
    requests \
    pyyaml \
    lama-cleaner

# Копируем все файлы в контейнер
WORKDIR /app
COPY . /app

# Экспонируем порты
EXPOSE 8501 8080 8081

# Запускаем три сервиса
CMD npx labelme-web --host 0.0.0.0 --port 8080 & \
    python -m lama_cleaner.api --host 0.0.0.0 --port 8081 & \
    streamlit run app.py --server.port 8501 --server.headless true
