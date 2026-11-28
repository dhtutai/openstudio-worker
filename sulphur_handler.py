"""RunPod worker using a private loopback ComfyUI process for Sulphur 3."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
import uuid
from sulphur_graph import build_graph, MODEL, TEXT, VIDEO_VAE, AUDIO_VAE

COMFY = Path(os.getenv('COMFY_ROOT', '/opt/ComfyUI'))
BASE = 'http://127.0.0.1:8188'
MAX_OUTPUT = 100 * 1024 * 1024
PROCESS = None


def api(path, body=None):
    request = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def start():
    global PROCESS
    from huggingface_hub import hf_hub_download
    cache = os.getenv('HF_HOME', '/runpod-volume/huggingface')
    for repo, revision, remote, folder in [
        ('SulphurAI/Sulphur-3', '988224b', MODEL, 'diffusion_models'),
        *[('Comfy-Org/MiniMax-H3', '4b53a35191437de05370da548e59aeabe2a5a09a', f'{folder}/{name}', folder) for folder, name in [('text_encoders', TEXT), ('vae', VIDEO_VAE), ('vae', AUDIO_VAE)]],
    ]:
        local = hf_hub_download(repo, remote, revision=revision, cache_dir=cache, token=os.getenv('HF_TOKEN'))
        dest = COMFY / 'models' / folder / Path(remote).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(local)
    PROCESS = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188', '--disable-auto-launch', '--disable-all-custom-nodes', '--disable-metadata'], cwd=COMFY)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if PROCESS.poll() is not None:
            raise RuntimeError('ComfyUI exited during startup.')
        try:
            info = api('/object_info')
            sample = {'model': 'sulphur3', 'prompt': 'test', 'width': 768, 'height': 768, 'seed': 1, 'num_frames': 124, 'fps': 24, 'steps': 20}
            missing = {v['class_type'] for v in build_graph(sample, 'test').values()} - info.keys()
            if missing:
                raise RuntimeError(f'Missing ComfyUI nodes: {sorted(missing)}')
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    raise RuntimeError('ComfyUI did not become ready.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def reference_photo(spec, target, supabase):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 40_000_000
    url = spec.get('url') if isinstance(spec, dict) else None
    if not isinstance(url, str) or not url.startswith(supabase + '/storage/v1/object/sign/'):
        raise ValueError('Invalid reference photo link.')
    with urllib.request.build_opener(NoRedirect).open(url, timeout=60) as response:
        raw = response.read(25 * 1024 * 1024 + 1)
    if len(raw) > 25 * 1024 * 1024:
        raise ValueError('Reference photo exceeds 25 MB.')
    image = Image.open(io.BytesIO(raw))
    if image.format not in {'PNG', 'JPEG', 'WEBP'}:
        raise ValueError('Reference must be PNG, JPEG or WebP.')
    ImageOps.exif_transpose(image).convert('RGB').save(target)


def handler(job):
    data = job.get('input', {})
    user_id, job_id = str(uuid.UUID(data['user_id'])), str(uuid.UUID(data['job_id']))
    prefix = f'openstudio/{job_id}'
    graph = build_graph(data, prefix)
    supabase = os.environ['SUPABASE_URL'].rstrip('/')
    key = os.environ['SUPABASE_SERVICE_ROLE_KEY']
    reference = COMFY / 'input' / f'{job_id}.png'
    output = None
    prompt_id = None
    try:
        if data.get('reference') is not None:
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference_photo(data['reference'], reference, supabase)
            graph = build_graph(data, prefix, reference.name)
        # Never retry a submission: an uncertain response might already be running.
        submitted = api('/prompt', {'prompt': graph, 'client_id': job_id})
        if submitted.get('node_errors') or not submitted.get('prompt_id'):
            raise RuntimeError('ComfyUI rejected the Sulphur workflow: ' + json.dumps(submitted)[:1000])
        prompt_id = submitted['prompt_id']
        deadline = time.monotonic() + int(os.getenv('SULPHUR_TIMEOUT_SECONDS', '3600'))
        while time.monotonic() < deadline:
            if PROCESS is not None and PROCESS.poll() is not None:
                raise RuntimeError('ComfyUI stopped during inference.')
            result = api('/history/' + prompt_id).get(prompt_id)
            if result:
                if result.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError('Sulphur inference failed: ' + json.dumps(result['status'].get('messages', []))[:1000])
                files = result.get('outputs', {}).get('save', {})
                candidates = [f for values in files.values() if isinstance(values, list) for f in values if isinstance(f, dict) and str(f.get('filename', '')).endswith('.mp4')]
                if not candidates:
                    raise RuntimeError('Sulphur did not return an MP4.')
                f = candidates[0]
                output = (COMFY / 'output' / f.get('subfolder', '') / f['filename']).resolve()
                output.relative_to((COMFY / 'output' / 'openstudio').resolve())
                if not output.name.startswith(job_id):
                    raise RuntimeError('Unexpected output filename.')
                break
            time.sleep(2)
        else:
            # A single job runs per worker; stop its process before any next job.
            if PROCESS is not None:
                PROCESS.kill()
            raise RuntimeError('Sulphur generation timed out; worker must restart.')
        if not output or not 0 < output.stat().st_size <= MAX_OUTPUT:
            raise RuntimeError('Invalid output or output exceeds 100 MB.')
        storage_path = f'{user_id}/{job_id}.mp4'
        request = urllib.request.Request(supabase + '/storage/v1/object/generations/' + storage_path, data=output.read_bytes(), method='POST', headers={'apikey': key, 'Authorization': 'Bearer ' + key, 'Content-Type': 'video/mp4', 'x-upsert': 'true'})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status not in (200, 201):
                raise RuntimeError('Private output upload failed.')
        return {'storage_path': storage_path, 'mime': 'video/mp4', 'seed': data['seed']}
    except Exception:
        # A lost submission/history response may leave inference running. Retire
        # the private process instead of accepting another overlapping job.
        if PROCESS is not None and PROCESS.poll() is None:
            PROCESS.kill()
        raise
    finally:
        reference.unlink(missing_ok=True)
        if output:
            output.unlink(missing_ok=True)
        if prompt_id:
            try:
                api('/history', {'delete': [prompt_id]})
            except Exception:
                pass


if __name__ == '__main__':
    import runpod
    start()
    runpod.serverless.start({'handler': handler})
