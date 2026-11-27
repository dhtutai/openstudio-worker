"""One model per RunPod queue worker. GPU inference + private Supabase Storage."""
import hashlib
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
from diffusers import AutoPipelineForImage2Image, AutoencoderKLWan, Flux2KleinPipeline, FluxPipeline, StableDiffusionXLPipeline, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video
from upscale import WEIGHTS_SHA256, WEIGHTS_URL, upscale

Image.MAX_IMAGE_PIXELS = 40_000_000  # Refuse decompression bombs from uploaded photos.

# MODEL_ARCH picks the model family. MODEL_KIND is kept for older endpoints: image -> flux, video -> wan.
KIND = os.getenv("MODEL_KIND", "image")
ARCH = os.getenv("MODEL_ARCH") or ("wan" if KIND == "video" else "flux")
DEFAULT_MODELS = {"upscale": "RealESRGAN_x4plus", "flux": "black-forest-labs/FLUX.1-schnell", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0", "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan22a14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "klein": "black-forest-labs/FLUX.2-klein-4B", "flux3": "black-forest-labs/flux-3-image-klein"}
if ARCH not in DEFAULT_MODELS:
    raise RuntimeError("MODEL_ARCH must be flux, flux3, sdxl, klein, wan, wan22, wan22a14b or upscale")
IS_VIDEO = ARCH in {"wan", "wan22", "wan22a14b"}
ANIMATES_PHOTO = {"wan22", "wan22a14b"}
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODELS[ARCH])
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HF_TOKEN = os.getenv("HF_TOKEN") or None
img2img = safety = None
PROMPT_UPSAMPLER = os.getenv("PROMPT_UPSAMPLER", "on") != "off"
SAFETY_THRESHOLD = float(os.getenv("SAFETY_THRESHOLD", "0.8"))
# FLUX 3 output sizes; the web server keeps the same table.
SIZED = {
    "standard": {"square": (1024, 1024), "landscape": (1344, 768), "portrait": (768, 1344), "wide": (1360, 768), "classic": (1248, 832), "ultrawide": (1568, 672)},
    "high": {"square": (1440, 1440), "landscape": (1888, 1088), "portrait": (1088, 1888), "wide": (1920, 1088), "classic": (1776, 1184), "ultrawide": (2208, 944)},
}
# Wan 2.2 14B: pixel count for each resolution, and the frame counts for 3, 5 and 8 seconds at 16 fps. Matches the web server.
WAN_AREAS = {"standard": (480, 832), "720p": (720, 1280)}
WAN_FRAMES = {49, 81, 129}
# FLUX.2 klein output sizes; the web server keeps the same table.
KLEIN_SIZED = {
    "standard": {"square": (1024, 1024), "landscape": (1344, 768), "portrait": (768, 1344)},
    "2k": {"square": (2048, 2048), "landscape": (2688, 1536), "portrait": (1536, 2688)},
}


def load_upscaler():
    """Download the Real-ESRGAN weights once, refusing anything but the expected file."""
    cache = Path(os.getenv("UPSCALER_CACHE", "/runpod-volume/upscaler"))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "RealESRGAN_x4plus.pth"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=300) as response:
            raw = response.read(100 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise RuntimeError("The upscaler download did not match its checksum.")
        partial = target.with_suffix(f".{os.getpid()}.part")
        partial.write_bytes(raw)
        partial.replace(target)
    return ModelLoader().load_from_file(target).model.to("cuda").eval()


if ARCH == "upscale":
    from spandrel import ModelLoader
    pipe = load_upscaler()
elif ARCH == "flux3":
    # Needs its own image (Dockerfile.flux3) and an HF_TOKEN whose account accepted the model's license.
    from diffusers import ComponentsManager, ModularPipeline
    from transformers import pipeline as transformers_pipeline
    manager = ComponentsManager()
    blocks = ModularPipeline.from_pretrained(MODEL_ID).blocks
    if not PROMPT_UPSAMPLER:
        blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(MODEL_ID, components_manager=manager)
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    manager.enable_auto_cpu_offload(device="cuda")  # The model and prompt upsampler do not fit on 48 GB together.
    # The license asks for content filtering or output review; this filters every output.
    safety = transformers_pipeline("image-classification", model="Falconsai/nsfw_image_detection", device="cpu")
elif ARCH == "flux":
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()  # Offload reduces VRAM needs; cold starts and CPU transfers add latency.
    pipe.vae.enable_tiling()
elif ARCH == "klein":
    # FLUX.2 klein 4B (Apache-2.0, open): one model for text-to-image and editing from reference photos.
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
elif ARCH == "sdxl":
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.to("cuda")  # SDXL fits on a 24 GB card without offload.
    img2img = AutoPipelineForImage2Image.from_pipe(pipe)  # Shares the same weights; used when a reference photo is given.
elif ARCH == "wan22a14b":
    # Wan 2.2 14B image-to-video: two 14B transformers take turns. Offload keeps only the active one on the GPU.
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
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


def matched_size(photo_width, photo_height, square):
    """The photo's shape with the pixel count of the given square size, on the 16-pixel grid."""
    area = square[0] * square[1]
    ratio = min(max(photo_width / photo_height, 1 / 3), 3)
    return tuple(max(256, round(side / 16) * 16) for side in ((area * ratio) ** 0.5, (area / ratio) ** 0.5))


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
    if ARCH == "upscale":
        try:
            source = fetch_reference(data.get("reference"))
            return run({"source": source, "seed": 0}, user_id, job_id)
        except torch.cuda.OutOfMemoryError:
            return {"error": "The GPU ran out of memory while upscaling. Try a smaller image."}
        except ValueError as error:  # A bad link or an unsupported size, explained to the user.
            return {"error": str(error)[:300]}
    sizes = {(480, 480), (832, 480), (480, 832)} if IS_VIDEO else {(1024, 1024), (1344, 768), (768, 1344)}
    width, height = data.get("width"), data.get("height")
    match_photo = ARCH == "flux3" and data.get("aspect") == "match"
    if ARCH == "flux3":
        quality = data.get("quality", "standard")
        if quality not in SIZED:
            return {"error": "Unsupported quality."}
        sizes = set(SIZED[quality].values())
    if ARCH == "klein":
        quality = data.get("quality", "standard")
        if quality not in KLEIN_SIZED:
            return {"error": "Unsupported quality."}
        sizes = set(KLEIN_SIZED[quality].values())
    if (width, height) not in sizes:
        return {"error": "Unsupported dimensions."}
    seed = data.get("seed", 0)
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        return {"error": "Invalid seed."}
    negative = data.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > 2000:
        return {"error": "Invalid negative prompt."}
    reference = strength = style = style_strength = None
    if ARCH in ANIMATES_PHOTO and data.get("reference") is None:
        return {"error": "This model animates a photo. Add a photo and try again."}
    if data.get("reference") is not None:
        if img2img is None and ARCH not in {"wan22", "wan22a14b", "klein", "flux3"}:
            return {"error": "This model cannot use a reference photo."}
        if img2img is not None:  # Only photo-to-image has a strength control.
            strength = data.get("strength", 0.7)
            if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0.1 <= strength <= 1:
                return {"error": "Strength must be between 0.1 and 1."}
        try:
            # Editing keeps the photo's own shape; everything else is cover-cropped to the requested size.
            reference = fetch_reference(data["reference"]) if ARCH in {"klein", "flux3", "wan22a14b"} else fetch_reference(data["reference"], width, height)
        except Exception as error:  # Reported to the user; the web server marks the job failed.
            return {"error": f"The reference photo could not be used: {str(error)[:300]}"}
    extra_references = []
    extra_specs = data.get("extra_references") or []
    if extra_specs:
        if ARCH != "klein" or reference is None or not isinstance(extra_specs, list) or len(extra_specs) > 3:
            return {"error": "FLUX.2 klein accepts up to 4 reference photos."}
        try:
            extra_references = [fetch_reference(spec) for spec in extra_specs]
        except Exception as error:
            return {"error": f"A reference photo could not be used: {str(error)[:300]}"}
    num_frames = 81
    if ARCH == "wan22a14b":  # The video keeps the photo's shape at the chosen pixel count; crop only the few pixels of rounding.
        quality, num_frames = data.get("quality", "standard"), data.get("num_frames", 81)
        if quality not in WAN_AREAS:
            return {"error": "Unsupported resolution."}
        if type(num_frames) is not int or num_frames not in WAN_FRAMES:
            return {"error": "Unsupported video length."}
        width, height = matched_size(reference.width, reference.height, WAN_AREAS[quality])
        reference = ImageOps.fit(reference, (width, height), Image.LANCZOS)
    if match_photo:
        if reference is None:
            return {"error": "Add a photo to match its shape."}
        width, height = matched_size(reference.width, reference.height, SIZED[quality]["square"])
    # Without an explicit size, klein returns at most about 1 MP when editing. At 2k, ask for the photo's shape at 4 MP.
    edit_size = None
    if ARCH == "klein" and reference is not None and quality == "2k":
        edit_size = matched_size(reference.width, reference.height, KLEIN_SIZED["2k"]["square"])
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
    params = dict(num_frames=num_frames, extra_references=extra_references, edit_size=edit_size, prompt=prompt, negative=negative, width=width, height=height, seed=seed, reference=reference, strength=strength, style=style, style_strength=style_strength)
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


class Blocked(Exception):
    pass


def generate(output, p):
    if ARCH == "upscale":
        upscale(pipe, p["source"]).save(output)
        return
    generator = torch.Generator("cpu").manual_seed(p["seed"])
    if ARCH == "flux":
        pipe(prompt=p["prompt"], width=p["width"], height=p["height"], generator=generator, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256).images[0].save(output)
    elif ARCH == "flux3":
        extra = {"image": p["reference"]} if p["reference"] is not None else {}
        image = pipe(prompt=p["prompt"], width=p["width"], height=p["height"], num_inference_steps=16, generator=generator, output="images", **extra)[0]
        if any(r["label"] == "nsfw" and r["score"] >= SAFETY_THRESHOLD for r in safety(image)):
            raise Blocked("The result was blocked by the content filter. Try a different prompt or photo.")
        image.save(output)
    elif ARCH == "klein":
        size = {} if p["reference"] is not None else {"height": p["height"], "width": p["width"]}  # With a photo, the output follows its shape.
        if p["edit_size"]:
            size = {"width": p["edit_size"][0], "height": p["edit_size"][1]}
        photos = [p["reference"], *p["extra_references"]] if p["reference"] is not None else None
        if p["extra_references"] and not size:  # With several photos, the output follows the shape of photo 1.
            width, height = matched_size(p["reference"].width, p["reference"].height, KLEIN_SIZED["standard"]["square"])
            size = {"width": width, "height": height}
        try:
            image = pipe(image=photos, prompt=p["prompt"], guidance_scale=1.0, num_inference_steps=4, generator=generator, **size).images[0]
        except torch.cuda.OutOfMemoryError as error:
            raise Blocked("The GPU ran out of memory. Use fewer reference photos or Standard quality.") from error
        image.save(output)
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
    elif ARCH == "wan22a14b":
        frames = pipe(image=p["reference"], prompt=p["prompt"], negative_prompt=p["negative"] or None, height=p["height"], width=p["width"], num_frames=p["num_frames"], num_inference_steps=40, guidance_scale=3.5, generator=generator).frames[0]
        export_to_video(frames, str(output), fps=16)
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
        try:
            generate(output, p)
        except Blocked as error:  # Reported to the user; the web server marks the job failed.
            return {"error": str(error)}
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
        used = {"references": 1 + len(p["extra_references"])} if p.get("extra_references") else {}
        return {"storage_path": key, "mime": mime, "seed": p["seed"], **used}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
