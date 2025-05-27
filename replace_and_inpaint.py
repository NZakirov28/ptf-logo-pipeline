import os, glob, requests
from pathlib import Path

# Создаём выходную папку
os.makedirs("inpaint_outputs", exist_ok=True)

# Для каждого изображения после сегментации
for img_path in sorted(glob.glob("runs/segment/predict/*.png")):
    name = Path(img_path).name
    out_path = f"inpaint_outputs/{name}"

    with open(img_path, "rb") as img_file:
        files = {
            "image": img_file,
            "mask": img_file  # маска совпадает с картинкой (альфа-канал)
        }

        try:
            response = requests.post("http://localhost:8081/inpaint", files=files, timeout=10)

            if response.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved: {out_path}")
            else:
                print(f"❌ Failed ({response.status_code}): {img_path}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending request for {img_path}: {e}")
