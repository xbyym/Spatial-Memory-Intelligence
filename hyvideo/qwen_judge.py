"""Four-device, greedy Qwen3.5 judge using the shared worldplay environment."""
import argparse
import base64
import json
import multiprocessing
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image

from hyvideo.utils import qwen_prompts as prompts
from hyvideo.utils.reproducibility import configure_qwen, audit_generation, record_runtime
from hyvideo.utils.qwen_results import decode_result


def load_model(model_path, device):
    configure_qwen()
    record_runtime("hy15_qwen")
    import torch
    import transformers.modeling_flash_attention_utils as flash
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    original = flash._is_packed_sequence
    def is_packed(position_ids, batch_size):
        if position_ids is not None and position_ids.ndim == 3:
            return False
        return original(position_ids, batch_size)
    flash._is_packed_sequence = is_packed
    global MODEL, PROCESSOR, TOKENIZER, DEVICE
    DEVICE = device
    MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation='flash_attention_2').eval()
    PROCESSOR = AutoProcessor.from_pretrained(model_path, max_pixels=576*28*28,
                                             min_pixels=16*28*28, padding_side='left')
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, padding_side='left')


def ready():
    return DEVICE


def judge(task, payload, max_new_tokens):
    import torch

    builders = dict(visual_quality=prompts._build_visual_quality_content,
                    create=prompts._build_create_content, sparse=prompts._build_sparse_content)
    if task == 'select_coverage':
        content = prompts._build_select_content(payload, prompts.COVERAGE, 'coverage_candidates')
    else:
        content = builders[task](payload)
    converted, images = [], []
    for item in content:
        if item['type'] == 'text':
            converted.append(item)
        else:
            with Image.open(item['url']) as image:
                buffer = BytesIO()
                image.convert('RGB').save(buffer, format='JPEG', quality=95)
            data = buffer.getvalue()
            converted.append(dict(type='image', image='data:image/jpeg;base64,' +
                                  base64.b64encode(data).decode()))
            images.append(Image.open(BytesIO(data)).convert('RGB'))
    system = ('You are a strict visual quality scorer.' if task == 'visual_quality'
              else 'You are a careful visual reasoning assistant.')
    messages = [[dict(role='system', content=system), dict(role='user', content=converted)]]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inputs = PROCESSOR(text=text, images=images, videos=None, padding=True,
                       return_tensors='pt').to(DEVICE)
    with torch.inference_mode():
        output = MODEL.generate(**inputs, eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.pad_token_id, do_sample=False,
            max_new_tokens=max_new_tokens, use_cache=True)
    audit_generation(inputs, output[0][inputs.input_ids.shape[-1]:], "hy15")
    raw = PROCESSOR.decode(output[0][inputs.input_ids.shape[-1]:],
                           skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    result = decode_result(task, raw, payload.get('coverage_candidates', []), max_selected=2)
    return dict(ok=True, task=task, raw=raw, result=result)


def worker_index(task, payload, worker_count=4, main_index=3):
    # Original runtime routes updates to replicas 0,1,2; visual quality stays on 3.
    if task == 'visual_quality':
        return main_index
    if task == 'sparse':
        branch = sum((i+1)*chunk for i, chunk in enumerate(payload['scenario_assumed_removed_chunk_ids']))
        key = payload['candidate_chunk_i']*1009 + payload['member_position']*97 + payload['member_from_end']*13 + branch
    elif task == 'select_coverage':
        key = payload['current_chunk_i']*17
    else:
        key = payload['cluster_id']
    update_workers = [i for i in range(worker_count) if i != main_index]
    return update_workers[key % len(update_workers)] if update_workers else main_index


def serve(model_path, port, devices=None, main_device=None):
    devices = devices or [f"cuda:{i}" for i in range(4)]
    main_device = main_device or devices[-1]
    if main_device not in devices or len(set(devices)) != len(devices):
        raise ValueError("main_device must occur exactly once in devices")
    main_index = devices.index(main_device)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    context = multiprocessing.get_context('spawn')
    workers = [ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=load_model,
               initargs=(model_path, device)) for device in devices]
    futures = [worker.submit(ready) for worker in workers]
    for future in futures:
        print('Loaded Qwen on', future.result(), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, dict(ok=True, ready=True))

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            task, payload = request['task'], request['payload']
            worker = workers[worker_index(task, payload, len(workers), main_index)]
            # Infrastructure errors remain fatal; invalid model decisions are skipped.
            try:
                result = worker.submit(judge, task, payload, request['max_new_tokens']).result()
            except Exception as exc:
                self.respond(500, dict(ok=False, error=str(exc)))
                return
            self.respond(200, result)

    try:
        ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
    finally:
        for worker in workers:
            worker.shutdown(cancel_futures=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=18870)
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3')
    parser.add_argument('--main-device')
    args = parser.parse_args()
    serve(args.model, args.port, args.devices.split(','), args.main_device)
