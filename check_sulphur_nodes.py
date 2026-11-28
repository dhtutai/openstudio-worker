"""Image build check: start ComfyUI on the CPU and confirm it has every node and required input the Sulphur graph uses."""
import io
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

from sulphur_graph import build_graph
from sulphur_handler import COMFY, api, reference_photo

def check_photo_crop():
    """A square photo made into a wide video must be cropped, so a circle stays a circle."""
    buffer = io.BytesIO()
    photo = Image.new("RGB", (1024, 1024), "white")
    ImageDraw.Draw(photo).ellipse((312, 312, 712, 712), fill="black")
    photo.save(buffer, "PNG")
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = buffer.getvalue()
    opener = mock.MagicMock()
    opener.open.return_value = response
    with tempfile.TemporaryDirectory() as temp, mock.patch("urllib.request.build_opener", return_value=opener):
        target = Path(temp) / "photo.png"
        reference_photo({"url": "https://x/storage/v1/object/sign/a"}, target, "https://x", (1344, 768))
        result = Image.open(target)
        left, top, right, bottom = result.convert("L").point(lambda v: 255 if v < 128 else 0).getbbox()
    if result.size != (1344, 768) or abs((right - left) - (bottom - top)) > 4:
        sys.exit(f"The reference photo is distorted: {result.size}, circle {right - left}x{bottom - top}")


check_photo_crop()
sample = {"model": "sulphur3", "prompt": "test", "width": 768, "height": 768, "seed": 1, "num_frames": 124, "fps": 24, "steps": 20}
process = subprocess.Popen([sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8188", "--disable-auto-launch", "--disable-all-custom-nodes"], cwd=COMFY)
try:
    deadline = time.monotonic() + 300
    while True:
        if process.poll() is not None or time.monotonic() > deadline:
            sys.exit("ComfyUI did not start.")
        try:
            info = api("/object_info")
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    problems = []
    for name, node in build_graph(sample, "check", "photo.png").items():
        known = info.get(node["class_type"])
        if known is None:
            problems.append(f"{name}: ComfyUI has no node {node['class_type']}")
            continue
        missing = set(known["input"].get("required", {})) - set(node["inputs"])
        if missing:
            problems.append(f"{name} ({node['class_type']}): missing required inputs {sorted(missing)}")
    if problems:
        sys.exit("\n".join(problems))
    print("ComfyUI has every node and required input the Sulphur graph uses.")
finally:
    process.kill()
