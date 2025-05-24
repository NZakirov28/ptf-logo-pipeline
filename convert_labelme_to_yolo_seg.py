import os, json, yaml
from pathlib import Path

def convert_all(json_dir, out_dir):
    img_dir = f"{out_dir}/images"
    lbl_dir = f"{out_dir}/labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for j in Path(json_dir).glob("*.json"):
        data = json.load(open(j))
        stem = j.stem
        # копируем изображение
        src_img = f"frames/{stem}.png"
        dst_img = f"{img_dir}/{stem}.png"
        os.system(f"cp {src_img} {dst_img}")

        # пишем .txt для YOLOv8 segmentation
        h, w = data['imageHeight'], data['imageWidth']
        lines = []
        for shape in data['shapes']:
            pts = shape['points']
            norm = []
            for x,y in pts:
                norm.extend([x/w, y/h])
            lines.append("0 " + " ".join(f"{v:.6f}" for v in norm))
        open(f"{lbl_dir}/{stem}.txt","w").write("\n".join(lines))

    # генерируем data.yaml
    cfg = {
        'path': '.',
        'train': img_dir,
        'val': img_dir,
        'nc': 1,
        'names': ['logo']
    }
    with open(f"{out_dir}/data.yaml","w") as f:
        yaml.dump(cfg, f)

if __name__=="__main__":
    convert_all("labelme/json","dataset")
