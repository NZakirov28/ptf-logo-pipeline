import os, glob, requests
from pathlib import Path

os.makedirs("inpaint_outputs", exist_ok=True)

# для каждого результата сегментации
for img in glob.glob("runs/segment/predict/*.png"):
    mask = img.replace(".png",".png").replace("runs/segment/predict","runs/segment/predict")  # маска тот же PNG
    files = {
        "image": open(img,"rb"),
        "mask": open(img,"rb")  # берём маску как альфа- слой; для точной настройки поправишь под txt
    }
    resp = requests.post("http://localhost:8081/inpaint", files=files)
    out_path = f"inpaint_outputs/{Path(img).name}"
    open(out_path,"wb").write(resp.content)
