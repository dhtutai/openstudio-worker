"""Image build check: start ComfyUI on the CPU and confirm it has every node and required input the Sulphur graph uses."""
import subprocess
import sys
import time
import urllib.error

from sulphur_graph import build_graph
from sulphur_handler import COMFY, api

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
