"""One model per RunPod queue worker. GPU inference + private Supabase Storage."""
import io
import os
import re
import tempfile
import uuid
import urllib.error
import urllib.request
from pathlib import Path
import runpod
from huggingface_hub import hf_hub_download
import torch
from PIL import Image, ImageOps
from diffusers import AutoPipelineForImage2Image, AutoencoderKLWan, FluxPipeline, StableDiffusionXLPipeline, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video

Image.MAX_IMAGE_PIXELS = 40_000_000  # Refuse decompression bombs from uploaded photos.

# MODEL_ARCH picks the model family. MODEL_KIND is kept for older endpoints: image -> flux, video -> wan.
KIND = os.getenv("MODEL_KIND", "image")
ARCH = os.getenv("MODEL_ARCH") or ("wan" if KIND == "video" else "flux")
DEFAULT_MODELS = {"flux": "black-forest-labs/FLUX.1-schnell", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0", "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}
if ARCH not in DEFAULT_MODELS:
    raise RuntimeError("MODEL_ARCH must be flux, sdxl, wan or wan22")
IS_VIDEO = ARCH in {"wan", "wan22"}
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODELS[ARCH])
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HF_TOKEN = os.getenv("HF_TOKEN") or None
img2img = None
if ARCH == "flux":
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # Offload reduces VRAM needs; cold starts and CPU transfers add latency.
    pipe.vae.enable_tiling()
elif ARCH == "sdxl":
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to("cuda")  # SDXL fits on a 24 GB card without offload.
    img2img = AutoPipelineForImage2Image.from_pipe(pipe)  # Shares the same weights; used when a reference photo is given.
elif ARCH == "wan22":
    # Wan 2.2 5B animates a photo (image-to-video only here); one pipeline keeps GPU offload simple on a 24 GB card.
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
else:
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

LORA_DIR = Path(os.getenv("LORA_CACHE", "/runpod-volume/loras"))
LORA_MAX_BYTES = int(os.getenv("LORA_MAX_MB", "500")) * 1024 * 1024
REFERENCE_MAX_BYTES = 25 * 1024 * 1024
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


def fetch_reference(spec, width=None, height=None):
    """Download the user's photo from a signed Supabase link; fit it to the output size when one is given."""
    url = spec.get("url") if isinstance(spec, dict) else None
    if not isinstance(url, str) or not url.startswith(f"{SUPABASE_URL}/storage/v1/object/sign/"):
        raise ValueError("Invalid reference photo link.")
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(REFERENCE_MAX_BYTES + 1)
    if len(raw) > REFERENCE_MAX_BYTES:
        raise ValueError("The reference photo is larger than 25 MB.")
    image = Image.open(io.BytesIO(raw))
    if image.format not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("The reference photo must be a PNG, JPG or WebP image.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (width, height), Image.LANCZOS) if width else image  # Cover-crop to the requested shape.


def handler(job):
    data = job.get("input", {})
    try:
        user_id, job_id = str(uuid.UUID(data["user_id"])), str(uuid.UUID(data["job_id"]))
    except (KeyError, ValueError, TypeError, AttributeError):
        return {"error": "A valid user ID and job ID are required."}
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        return {"error": "Prompt must contain 1–2000 characters."}
    if data.get("model") != ARCH:
        return {"error": "This endpoint is configured for a different model."}
    sizes = {(480, 480), (832, 480), (480, 832)} if IS_VIDEO else {(1024, 1024), (1344, 768), (768, 1344)}
    width, height = data.get("width"), data.get("height")
    if (width, height) not in sizes:
        return {"error": "Unsupported dimensions."}
    seed = data.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        return {"error": "Invalid seed."}
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > 2000:
        return {"error": "Invalid negative prompt."}
    reference = strength = style = style_strength = None
    if ARCH == "wan22" and data.get("reference") is None:
        return {"error": "This model animates a photo. Add a photo and try again."}
    if data.get("reference") is not None:
        if img2img is None and ARCH != "wan22":
            return {"error": "This model cannot use a reference photo."}
        if img2img is not None:  # Only photo-to-image has a strength control.
            strength = data.get("strength", 0.7)
            if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0.1 <= strength <= 1:
                return {"error": "Strength must be between 0.1 and 1."}
        try:
            reference = fetch_reference(data["reference"], width, height)
        except Exception as error:  # Reported to the user; the web server refunds the credits.
            return {"error": f"The reference photo could not be used: {str(error)[:300]}"}
    if data.get("style") is not None:
        if ARCH != "sdxl":
            return {"error": "This model cannot use a style reference."}
        style_strength = data.get("style_strength", 0.6)
        if isinstance(style_strength, bool) or not isinstance(style_strength, (int, float)) or not 0.1 <= style_strength <= 1:
            return {"error": "Style strength must be between 0.1 and 1."}
        try:
            style = fetch_reference(data["style"])
        except Exception as error:
            return {"error": f"The style reference could not be used: {str(error)[:300]}"}
    lora_path = lora_strength = None
    if data.get("lora") is not None:
        try:
            lora_path, lora_strength = fetch_lora(data["lora"])
        except Exception as error:
            return {"error": f"LoRA could not be loaded: {str(error)[:300]}"}
    params = dict(prompt=prompt, negative=negative, width=width, height=height, seed=seed, reference=reference, strength=strength, style=style, style_strength=style_strength)
    try:
        if lora_path:
            try:
                pipe.load_lora_weights(lora_path, adapter_name="user")
                pipe.set_adapters(["user"], adapter_weights=[lora_strength])
            except Exception as error:  # Usually a LoRA trained for a different architecture.
                return {"error": f"This LoRA could not be applied to the model: {str(error)[:300]}"}
        return run(params, user_id, job_id)
    finally:
        if lora_path:
            pipe.unload_lora_weights()  # Never let one user's adapter leak into the next job.


def generate(output, p):
    generator = torch.Generator("cpu").manual_seed(p["seed"])
    if ARCH == "flux":
        pipe(prompt=p["prompt"], width=p["width"], height=p["height"], generator=generator, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256).images[0].save(output)
    elif ARCH == "sdxl":
        target = img2img if p["reference"] is not None else pipe
        extra = {}
        if p["style"] is not None:  # IP-Adapter: borrow the look of the style photo. Loaded per job, then removed.
            target.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors", image_encoder_folder="sdxl_models/image_encoder")
            target.set_ip_adapter_scale(p["style_strength"])
            extra["ip_adapter_image"] = p["style"]
        try:
            if p["reference"] is not None:
                image = img2img(prompt=p["prompt"], negative_prompt=p["negative"] or None, image=p["reference"], strength=p["strength"], num_inference_steps=30, guidance_scale=6.0, generator=generator, **extra).images[0]
            else:
                image = pipe(prompt=p["prompt"], negative_prompt=p["negative"] or None, width=p["width"], height=p["height"], num_inference_steps=30, guidance_scale=6.0, generator=generator, **extra).images[0]
        finally:
            if p["style"] is not None:
                target.unload_ip_adapter()
        image.save(output)
    elif ARCH == "wan22":
        frames = pipe(image=p["reference"], prompt=p["prompt"], negative_prompt=p["negative"] or None, height=p["height"], width=p["width"], num_frames=81, num_inference_steps=30, guidance_scale=5.0, generator=generator).frames[0]
        export_to_video(frames, str(output), fps=24)
    else:
        frames = pipe(prompt=p["prompt"], width=p["width"], height=p["height"], generator=generator, negative_prompt=p["negative"], num_frames=81, num_inference_steps=30, guidance_scale=5.0).frames[0]
        export_to_video(frames, str(output), fps=16)


def run(p, user_id, job_id):
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / ("output.mp4" if IS_VIDEO else "output.png")
        mime = "video/mp4" if IS_VIDEO else "image/png"
        generate(output, p)
        key = f"{user_id}/{job_id}{output.suffix}"
        if output.stat().st_size > 104857600:
            return {"error": "Generated file exceeds the 100 MB storage limit."}
        request = urllib.request.Request(
            f"{SUPABASE_URL}/storage/v1/object/generations/{key}",
            data=output.read_bytes(), method="POST",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": mime, "x-upsert": "true"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status not in (200, 201):
                    raise RuntimeError("Private output upload failed")
        except urllib.error.HTTPError as error:  # Include Supabase's explanation (e.g. a missing bucket) in the job error.
            raise RuntimeError(f"Output upload failed: HTTP {error.code} {error.read(300).decode('utf-8', 'replace')}") from error
        return {"storage_path": key, "mime": mime, "seed": p["seed"]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
