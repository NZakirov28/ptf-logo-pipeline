import streamlit as st
import subprocess, os
from ffmpeg_split import split_video
from convert_labelme_to_yolo_seg import convert_all

st.title("🚀 PTF Logo Pipeline")

# 1. Загрузка видео
video = st.file_uploader("1️⃣ Загрузить видео (MP4)", type="mp4")
if video:
    with open("input.mp4","wb") as f: f.write(video.getbuffer())
    st.success("Видео сохранено как input.mp4")
    if st.button("2️⃣ Разбить на кадры (1 fps)"):
        split_video("input.mp4", fps=1, out_dir="frames")
        st.image("frames/frame_0001.png", caption="Пример кадра", width=200)

# 2. Разметка в LabelMe
st.markdown("### 3️⃣ Разметка логотипов")
st.markdown("[Открыть LabelMe Web →](http://localhost:8080/)")

# 3. Конвертация в YOLOv8
if st.button("4️⃣ Конвертировать разметку → YOLOv8"):
    convert_all("labelme/json", "dataset")
    st.success("Готов dataset/ с images/ и labels/ и data.yaml")

# 4. Обучение
if st.button("5️⃣ Train YOLOv8 (30 эпок)"):
    cmd = "yolo task=segment mode=train data=dataset/data.yaml model=/models/yolov8n-seg.pt epochs=30 imgsz=640"
    st.text(cmd)
    subprocess.run(cmd.split())
    st.success("Обучение завершено")

# 5. Инференс + Inpainting
if st.button("6️⃣ Predict + Inpaint"):
    # сегментация
    cmd = "yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=frames save=True"
    st.text(cmd); subprocess.run(cmd.split())
    # inpaint
    subprocess.run(["python","replace_and_inpaint.py"])
    st.success("Inpainting готов — см inpaint_outputs/")
    # показать один результат
    st.image("inpaint_outputs/frame_0001.png", caption="Результат", width=300)
