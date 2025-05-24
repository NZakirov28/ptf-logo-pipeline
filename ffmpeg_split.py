import subprocess, os

def split_video(input_path, fps=1, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-vf", f"fps={fps}",
        f"{out_dir}/frame_%04d.png"
    ])
