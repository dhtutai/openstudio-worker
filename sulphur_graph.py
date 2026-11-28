"""Fixed H3 graph: no client-supplied nodes, model paths, or output paths.

Based on Comfy-Org/workflow_templates/benchmarks/video_minimax_h3_t2v.json.
The diffusion model is always Sulphur's s3.safetensors, never base H3.
"""
SIZES = {'square': (768, 768), 'landscape': (1344, 768), 'portrait': (768, 1344)}
MODEL = 's3.safetensors'
TEXT = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
VIDEO_VAE = 'minimax_h3_video_vae_int8_convrot.safetensors'
AUDIO_VAE = 'minimax_h3_audio_vae_fp32.safetensors'


def build_graph(data, prefix, reference=None):
    if data.get('model') != 'sulphur3':
        raise ValueError('This endpoint only serves Sulphur 3.')
    if not isinstance(data.get('prompt'), str) or not 1 <= len(data['prompt'].strip()) <= 2000:
        raise ValueError('Prompt must contain 1–2000 characters.')
    if (data.get('width'), data.get('height')) not in SIZES.values():
        raise ValueError('Unsupported dimensions.')
    if type(data.get('seed')) is not int or not 0 <= data['seed'] <= 2147483647:
        raise ValueError('Invalid seed.')
    if data.get('negative_prompt'):
        raise ValueError('Sulphur 3 does not support negative prompts.')
    if any(data.get(k) is not None for k in ('lora', 'style', 'strength', 'style_strength')):
        raise ValueError('Sulphur 3 does not support these adapters or strength controls.')
    if data.get('num_frames') != 124 or data.get('fps') != 24 or data.get('steps') != 20:
        raise ValueError('Sulphur 3 requires 124 frames, 24 fps and 20 steps.')
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    graph = {
        'model': node('UNETLoader', unet_name=MODEL, weight_dtype='default'),
        'clip': node('CLIPLoader', clip_name=TEXT, type='minimax', device='default'),
        'vae': node('VAELoader', vae_name=VIDEO_VAE),
        'audio_vae': node('VAELoader', vae_name=AUDIO_VAE),
        'condition': node('MiniMaxH3ImageToVideo', clip=['clip', 0], vae=['vae', 0], prompt=data['prompt'], width=data['width'], height=data['height'], length=124),
        'noise': node('RandomNoise', noise_seed=data['seed']),
        'guider': node('BasicGuider', model=['model', 0], conditioning=['condition', 0]),
        'sampler': node('KSamplerSelect', sampler_name='res_multistep'),
        'schedule': node('BasicScheduler', model=['model', 0], scheduler='simple', steps=20, denoise=1),
        'sample': node('SamplerCustomAdvanced', noise=['noise', 0], guider=['guider', 0], sampler=['sampler', 0], sigmas=['schedule', 0], latent_image=['condition', 1]),
        'frames': node('VAEDecode', samples=['sample', 0], vae=['vae', 0]),
        'audio': node('VAEDecodeAudio', samples=['sample', 0], vae=['audio_vae', 0]),
        'video': node('CreateVideo', images=['frames', 0], audio=['audio', 0], fps=24),
        'save': node('SaveVideo', video=['video', 0], filename_prefix=prefix, format='mp4', **{'format.codec': 'h264', 'format.codec.encoding': 'auto', 'codec': 'h264'}),
    }
    if reference:
        graph['reference'] = node('LoadImage', image=reference)
        graph['condition']['inputs']['first_frame'] = ['reference', 0]
    return graph
