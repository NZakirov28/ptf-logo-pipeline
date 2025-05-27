import streamlit as st
import subprocess, os, glob
from ffmpeg_split import split_video
from convert_labelme_to_yolo_seg import convert_all
from pathlib import Path

st.set_page_config(page_title="PTF Logo Pipeline", layout="wide")
st.title("🚀 PTF Logo Pipeline")

# --- Инициализация session_state ---
if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False
if "frames_extracted" not in st.session_state:
    st.session_state.frames_extracted = False
if "dataset_ready" not in st.session_state:
    st.session_state.dataset_ready = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "inpaint_done" not in st.session_state:
    st.session_state.inpaint_done = False

# --- Шаг 1: Загрузка видео ---
video = st.file_uploader("1️⃣ Загрузить видео (MP4)", type="mp4")
if video:
    with open("input.mp4", "wb") as f:
        f.write(video.getbuffer())
    st.session_state.video_uploaded = True
    st.success("✅ Видео сохранено как input.mp4")

# --- Шаг 2: Разбивка на кадры ---
if st.session_state.video_uploaded:
    if st.button("2️⃣ Разбить на кадры (1 кадр/сек)"):
        with st.spinner("Извлечение кадров..."):
            split_video("input.mp4", fps=1, out_dir="frames")
            st.session_state.frames_extracted = True
        st.success("📸 Кадры извлечены в папку frames/")
        st.image("frames/frame_0001.png", caption="Пример кадра", width=200)

# --- Шаг 3: Разметка логотипов ---
if st.session_state.frames_extracted:
    st.markdown("### 3️⃣ Разметка логотипов")
    st.markdown("[🔗 Открыть LabelMe Web](http://localhost:8080/)")

# --- Шаг 4: Конвертация разметки в YOLOv8 формат ---
if st.session_state.frames_extracted:
    if st.button("4️⃣ Конвертировать разметку → YOLOv8"):
        with st.spinner("Конвертация..."):
            convert_all("labelme/json", "dataset")
            st.session_state.dataset_ready = True
        st.success("✅ Dataset готов в папке `dataset/`")

# --- Шаг 5: Обучение YOLOv8 ---
if st.session_state.dataset_ready:
    if st.button("5️⃣ Обучить YOLOv8 (30 эпох)"):
        model_path = "/models/yolov8n-seg.pt"
        if not Path(model_path).exists():
            st.error(f"❌ Модель не найдена: {model_path}")
        else:
            with st.spinner("Обучение YOLO..."):
                cmd = f"yolo task=segment mode=train data=dataset/data.yaml model={model_path} epochs=30 imgsz=640"
                st.code(cmd)
                subprocess.run(cmd.split())
                st.session_state.model_trained = True
            st.success("🎯 Обучение завершено")

# --- Шаг 6: Инференс + Inpainting ---
if st.session_state.model_trained:
    if st.button("6️⃣ Предсказать + Inpaint"):
        weight_path = "runs/segment/train/weights/best.pt"
        if not Path(weight_path).exists():
            st.error("❌ Не найден файл модели: runs/segment/train/weights/best.pt")
        else:
            with st.spinner("🧠 Запуск YOLOv8..."):
                cmd = f"yolo task=segment mode=predict model={weight_path} source=frames save=True"
                st.code(cmd)
                subprocess.run(cmd.split())

            with st.spinner("🪄 Запуск inpainting..."):
                subprocess.run(["python", "replace_and_inpaint.py"])
                st.session_state.inpaint_done = True
            st.success("🧽 Inpainting завершён — смотри результаты ниже")

# --- Показываем примеры результата ---
if st.session_state.inpaint_done:
    st.markdown("### 📊 Результаты inpainting")
    result_imgs = sorted(glob.glob("inpaint_outputs/*.png"))
    if result_imgs:
        cols = st.columns(3)
        for i, img_path in enumerate(result_imgs[:9]):
            with cols[i % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
    else:
        st.warning("Нет обработанных кадров для показа.")
