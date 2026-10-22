---
title: paddle-ocr-wasm
short_description: PaddleOCR-VL-1.6 document reading in the browser
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - PaddlePaddle/PaddleOCR-VL-1.6
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Document reading with PaddlePaddle's [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
(Apache-2.0), computed on your own CPU in the browser: the model, compiled to WebAssembly with
fusor-ai, downloads once and then runs sealed from the network, reading the text, a table, a
formula, a chart or a seal off an image you pick. The source is
[github.com/fusor-rs/paddle-ocr-wasm](https://github.com/fusor-rs/paddle-ocr-wasm).
