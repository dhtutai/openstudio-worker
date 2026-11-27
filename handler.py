"""One model per RunPod queue worker. GPU inference + private Supabase Storage."""
import os
import re
import tempfile
import uuid
import urllib.request
from pathlib import Path
import runpod
from huggingface_hub import hf_hub_download
import torch
from diffusers import FluxPipeline, WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

KIND = os.getenv("MODEL_KIND", "image")
if KIND not in {"image", "video"}:
    raise RuntimeError("MODEL_KIND must be image or video")
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell" if KIND == "image" else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
if KIND == "image":
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
else:
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
# Offload reduces VRAM requirements; cold starts and CPU transfers add latency.
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

HF_TOKEN = os.getenv("HF_TOKEN") or None
LORA_DIR = Path(os.getenv("LORA_CACHE", "/runpod-volume/loras"))
LORA_MAX_BYTES = int(os.getenv("LORA_MAX_MB", "500")) * 1024 * 1024
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FILE_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}\.safetensors$")
REV_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def fetch_lora(spec):
    """Return a local .safetensors path for a validated LoRA spec, downloading on first use."""
    if not isinstance(spec, dict):
        raise ValueError("Invalid LoRA.")
    strength = spec.get("strength", 1.0)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0 <= strength <= 2:
        raise ValueError("Invalid LoRA strength.")
    key = spec.get("key")
    if not isinstance(key, str) or not UUID_RE.match(key):
        raise ValueError("Invalid LoRA.")
    if spec.get("source") == "huggingface":
        repo, filename, revision = spec.get("repo"), spec.get("file"), spec.get("revision")
        if not (isinstance(repo, str) and REPO_RE.match(repo) and isinstance(filename, str) and FILE_RE.match(filename)
                and ".." not in filename.split("/") and isinstance(revision, str) and REV_RE.match(revision)):
            raise ValueError("Invalid Hugging Face LoRA reference.")
        # Pinned to a commit, so the cached file can never silently change.
        return hf_hub_download(repo_id=repo, filename=filename, revision=revision, token=HF_TOKEN), float(strength)
    if spec.get("source") == "upload":
        url = spec.get("url")
        if not isinstance(url, str) or not url.startswith(f"{SUPABASE_URL}/storage/v1/object/sign/loras/"):
            raise ValueError("Invalid LoRA download link.")
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        target = LORA_DIR / f"{key}.safetensors"  # The key is the unique LoRA ID; uploads are never overwritten.
        if not target.exists():
            partial = target.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=300) as response, open(partial, "wb") as out:
                copied = 0
                while chunk := response.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > LORA_MAX_BYTES:
                        partial.unlink(missing_ok=True)
                        raise ValueError("LoRA file is too large.")
                    out.write(chunk)
            partial.replace(target)
        return str(target), float(strength)
    raise ValueError("Unknown LoRA source.")


def handler(job):
    data = job.get("input", {})
    try:
        user_id, job_id = str(uuid.UUID(data["user_id"])), str(uuid.UUID(data["job_id"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        return {"error": "A valid user ID and job ID are required."}
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        return {"error": "Prompt must contain 1–2000 characters."}
    if data.get("model") != ("flux" if KIND == "image" else "wan"):
        return {"error": "This endpoint is configured for a different model kind."}
    sizes = {(1024, 1024), (1344, 768), (768, 1344)} if KIND == "image" else {(480, 480), (832, 480), (480, 832)}
    width, height = data.get("width"), data.get("height")
    if (width, height) not in sizes:
        return {"error": "Unsupported dimensions."}
    seed = data.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        return {"error": "Invalid seed."}
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > 2000:
        return {"error": "Invalid negative prompt."}
    lora_path = lora_strength = None
    if data.get("lora") is not None:
        try:
            lora_path, lora_strength = fetch_lora(data["lora"])
        except Exception as error:  # Reported to the user; the job is refunded by the web server.
            return {"error": f"LoRA could not be loaded: {str(error)[:300]}"}
    options = dict(prompt=prompt, width=width, height=height, generator=torch.Generator("cpu").manual_seed(seed))
    try:
        if lora_path:
            try:
                pipe.load_lora_weights(lora_path, adapter_name="user")
                pipe.set_adapters(["user"], adapter_weights=[lora_strength])
            except Exception as error:  # Usually a LoRA trained for a different architecture.
                return {"error": f"This LoRA could not be applied to the model: {str(error)[:300]}"}
        return run(options, user_id, job_id, seed, negative)
    finally:
        if lora_path:
            pipe.unload_lora_weights()  # Never let one user's adapter leak into the next job.


def run(options, user_id, job_id, seed, negative):
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
        if KIND == "image":
            output = Path(temp) / "output.png"
            pipe(**options, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256).images[0].save(output)
            mime = "image/png"
        else:
            output = Path(temp) / "output.mp4"
            frames = pipe(**options, negative_prompt=negative, num_frames=81, num_inference_steps=30, guidance_scale=5.0).frames[0]
            export_to_video(frames, str(output), fps=16)
            mime = "video/mp4"
        key = f"{user_id}/{job_id}{output.suffix}"
        if output.stat().st_size > 104857600:
            return {"error": "Generated file exceeds the 100 MB storage limit."}
        request = urllib.request.Request(
            f"{SUPABASE_URL}/storage/v1/object/generations/{key}",
            data=output.read_bytes(), method="POST",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": mime, "x-upsert": "true"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status not in (200, 201):
                raise RuntimeError("Private output upload failed")
        return {"storage_path": key, "mime": mime, "seed": seed}



if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
