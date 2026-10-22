# Benchmarks

How fast PaddleOCR-VL-1.6 reads on one machine, in every runtime this project has: the page in
Chrome (eight threads and one), the native `paddle-ocr` CLI, the single-threaded runtimes in
Node, and the PyTorch reference on the CPU. Everything here was measured on 2026-10-21 on the
machine below; nothing is quoted from elsewhere.

Reading an image has two phases, and the tables give both. **First token** is the time from the
image to the answer's first token: the preprocessing, the vision tower over every 14×14 patch
and the prompt through the language model. It grows with the image: a text line is 600–700
patches, a page up to 5120. **Tokens per second** is the speed of the answer after that first
token, one greedy token at a time. **CER** is the character error rate of the answer against
the reference's on the same image (0 is identical).

## The machine

| | |
|---|---|
| Computer | Apple M4 Pro, 14 cores (10 performance, 4 efficiency), 24 GB |
| OS | macOS 27.0.1 (Darwin 27.0.0) |
| Native CLI | `paddle-ocr` 0.1.0, Rust 1.99.0, release build |
| Browser | Chrome headless shell (Playwright 1.64.1), Node 22.16.0 |
| PyTorch | torch 2.14.1, transformers 5.18.0, f32, the golden harness's reference (`benchmarks/pytorch.py`) |
| Bundle | `mix`, 831 MB: 4-bit vision layers, everything else 7-bit |

Other work ran on the machine throughout (1-minute load average 4–6 on 14 cores), which slows
the spinning thread team more than PyTorch's; every number here is one run, so read them as their
order of magnitude. Rerun them with `just e2e` (add `--threads 1`), `just smoke`,
`target/release/paddle-ocr read`, and `just pytorch`.

## The images

PaddleOCR's own test files (Apache-2.0), each read with its task (`fixtures/tasks.json`):

| Image | Task | Size | Resized to | Patches | Image tokens |
|---|---|---:|---:|---:|---:|
| `formula.png` | formula | 448×64 | 896×140 | 640 | 160 |
| `table.jpg` | table | 551×132 | 700×168 | 600 | 150 |
| `seal.png` | seal | 640×640 | 644×644 | 2116 | 529 |
| `book.jpg` | text | 1100×708 | 1092×700 | 3900 | 975 |
| `textline.png` | text | 514×64 | 980×140 | 700 | 175 |

## In Chrome

`just e2e` (threaded, relaxed SIMD, eight compute threads) and `just e2e --threads 1`
(single-threaded, relaxed SIMD). The first image is read while the model loads, so its first
token includes the end of the load; the table row says so.

| Image | Threads | First token | Tokens per second | Tokens | CER |
|---|---:|---:|---:|---:|---:|
| table | 8 | 3.87 s (with the load) | 348 | 71 | 0 |
| formula | 8 | 0.92 s | 371 | 79 | 0 |
| book | 8 | 9.31 s | 272 | 302 | 0.132 |
| table | 1 | 8.56 s (with the load) | 112 | 71 | 0 |
| formula | 1 | 6.35 s | 112 | 79 | 0 |
| book | 1 | 67.43 s | 59 | 302 | 0.132 |

Every run made no network request after the model was ready, and the revisit loaded the 831 MB
bundle from the browser's storage in about a second.

## Native, against PyTorch

`target/release/paddle-ocr read dist/model <image> --task <task> --threads <n>`, and
`just pytorch 8` (`torch.set_num_threads(8)`; the first token and the last timed in the same
`generate`, after a warm-up read).

| Image | Runtime | Threads | First token | Tokens per second |
|---|---|---:|---:|---:|
| formula | paddle-ocr-wasm, native | 8 | 0.68 s | 409 |
| formula | PyTorch, f32 | 8 | 0.72 s | 61 |
| seal | paddle-ocr-wasm, native | 8 | 2.85 s | 363 |
| seal | PyTorch, f32 | 8 | 2.34 s | 60 |
| book | paddle-ocr-wasm, native | 8 | 6.70 s | 287 |
| book | PyTorch, f32 | 8 | 4.85 s | 58 |
| formula | paddle-ocr-wasm, native | 1 | 4.49 s | 120 |
| seal | paddle-ocr-wasm, native | 1 | 19.71 s | 92 |
| book | paddle-ocr-wasm, native | 1 | 47.03 s | 71 |

The answer is where fusor-ai wins: 5–7× PyTorch's tokens per second, because each token is a
handful of 7-bit matrix-vector products with no framework in between. The first token is where
it loses on large images: PyTorch's f32 matrix products go through Apple's Accelerate
(`BLAS_INFO=accelerate`), which runs them on the M4's matrix units, and on the 2116- and
3900-patch images they finish the vision tower 1.2–1.4× sooner than fusor-ai's int8 SIMD
kernels, which are written for WebAssembly first. On small crops the two are even. PyTorch has
no browser build.

## In Node

`just smoke`: the single-threaded runtimes, on the PNG images.

| Build | Image | First token | Tokens per second | CER |
|---|---|---:|---:|---:|
| SIMD128 | textline | 15.73 s | 85 | 0 |
| SIMD128 | formula | 11.62 s | 95 | 0 |
| SIMD128 | seal | 48.30 s | 73 | 0 |
| relaxed SIMD | textline | 8.60 s | 97 | 0 |
| relaxed SIMD | formula | 6.39 s | 111 | 0 |
| relaxed SIMD | seal | 28.51 s | 84 | 0 |

## Agreement with the reference

`just parity` feeds each stage the reference's own input, with the f32 bundle, on `textline.png`:

| Stage | Relative L2 | Max abs |
|---|---:|---:|
| Preprocessing (Pillow-exact resize, patches) | 0 | 0 |
| Patch embedding and position grid | 3.9·10⁻⁷ | 6.0·10⁻⁵ |
| Vision encoder (27 layers) and final norm | 1.1·10⁻² | 14 |
| Projector | 1.9·10⁻⁶ | 2.2·10⁻⁵ |
| Prompt ids and multimodal rotary positions | identical | |
| Prefill: last hidden state | 6.1·10⁻⁶ | 1.6·10⁻⁴ |
| Prefill: logits | 2.5·10⁻⁶ | 7.3·10⁻⁵ |
| Greedy answer | 7 of 7 tokens identical | |

The vision encoder's attention runs on fusor-ai's int8 kernels (`blocks::attention`: keys and
values quantized per head, the keys' mean subtracted first), which is where its 1% comes from;
with an f32 attention the stage agrees within 4·10⁻⁶, and the answers on every image below are
the same either way, while the int8 attention reaches a page's first token about a third
sooner.

Quantized, against the reference's answers (native CLI):

| Bundle | Size | textline | table | formula | seal | book |
|---|---:|---:|---:|---:|---:|---:|
| f32 | 3.6 GB | | | | | 0.014 |
| `q7`: every large matrix 7-bit | 1037 MB | 0 | 0 | 0 | 0 | 0.116 |
| `mix`: vision layers 4-bit | 831 MB | 0 | 0 | 0 | 0 | 0.132 |
| language model layers 4-bit too | 703 MB | 0 | 0 | 0.023 | 0 | 4.74 |

The book page is a photograph of a two-page spread; greedy reading of it passes near-ties where
any change in rounding picks another word and the text goes on differently from there, which is
why even the f32 bundle differs from the reference by 1.4%. The answers stay readable: the
first sentence of `q7`'s matches the printed page more closely than the reference's. The
language model does not take 4 bits: it falls into repeating itself, so `mix` keeps it at 7.
