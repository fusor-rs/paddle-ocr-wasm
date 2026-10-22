"""The reference's speed on the CPU, measured as the page measures itself: the time from the image
to the answer's first token (preprocessing, the vision tower and the prompt), and the tokens per
second after it, greedy, with the KV cache. The reference is the golden harness's: PaddleOCR-VL-1.6
in transformers, f32, its Pillow image processor, the position grid resampled as PaddlePaddle does.

    python pytorch.py <checkpoint dir> <images dir> <tasks.json> <threads> <image>...
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor, StoppingCriteria

PROMPTS = {"ocr": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:", "chart": "Chart Recognition:", "seal": "Seal Recognition:"}
MAX_PIXELS = 1280 * 28 * 28
MAX_NEW_TOKENS = 2048

checkpoint, images, tasks, threads, *names = sys.argv[1:]
tasks = json.loads(Path(tasks).read_text())
torch.set_num_threads(int(threads))
model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()
model.model.visual.vision_model.embeddings.interpolation_align_corners = False
processor = AutoProcessor.from_pretrained(checkpoint)
processor.image_processor = AutoImageProcessor.from_pretrained(checkpoint, backend="pil")
size = {"shortest_edge": processor.image_processor.size.shortest_edge, "longest_edge": MAX_PIXELS}


class Clock(StoppingCriteria):
    """The time of every generated token; never stops the generation."""

    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores, **kwargs):
        self.times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool)


def read(image, task):
    """Seconds from the image to the first token and to the last, and the tokens written."""
    start = time.perf_counter()
    clock = Clock()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPTS[task]}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"size": size})
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True, stopping_criteria=[clock])
    return clock.times[0] - start, clock.times[-1] - start, len(clock.times)


for name in names:
    image = Image.open(next(Path(images).glob(f"{name}.*")))
    read(image, tasks[name])
    first, total, tokens = read(image, tasks[name])
    print(f"{name}: threads {threads}, first token {first:.2f}s, {tokens} tokens in {total:.2f}s ({(tokens - 1) / (total - first):.1f}/s after the first)", flush=True)
