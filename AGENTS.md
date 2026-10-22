# AGENTS.md

Rules for coding agents in this repository that are its own. fusor-ai's
`vendor/fusor-ai/AGENTS.md` is the standard and binds here too (`CLAUDE.md`
imports both); this file adds what is particular to this product.

paddle-ocr-wasm is the PaddleOCR-VL-1.6 (PaddlePaddle/PaddleOCR-VL-1.6) document
reader: the model, a 0.9B vision-language model (a NaViT vision encoder and
ERNIE 4.5 0.3B), running in the browser on the CPU, sealed from the network,
reading the text, a table, a formula, a chart or a seal off an image. The README
describes what it does; don't invent scope, features or options beyond the task.

## Repository

- `vendor/fusor-ai/models/paddle-ocr/runtime`: the wasm-bindgen surface of the
  model (bundle parts and RGBA pixels in, the answer's text out), built as four
  browser runtimes.
- `vendor/fusor-ai/models/paddle-ocr/cli`: the native CLI `paddle-ocr`
  (`compile`, `read`, `bench`, `parity`).
- `app/`: the page, a [fusor](https://github.com/fusor-rs/fusor) app with its own
  Cargo workspace (fusor pins wasm-bindgen 0.2.117, the runtime 0.2.129): the
  image, the task, reading and the live measurements as Rust signals in
  `src/app.rs`, the OTSL table view in `src/table.rs`, the template in
  `web/index.html`, the engine bridge and the image decoding in `web/app.js`,
  and in `engine/` the engine worker, beside the runtimes `just runtimes`
  builds.
- `scripts/`: the checkpoint and image downloads (pinned, sha256-checked), the
  Node smoke test and the character error rate; `e2e/run.mjs`: the headless
  Chrome run; `fixtures/`: each image's task and the reference's answers.
- [fusor-ai](https://github.com/fusor-rs/fusor-ai): the model (`paddle-ocr`),
  its compiler (`paddle-ocr-compile`), the kernels, blocks and bundle format
  come from `vendor/fusor-ai`, a git submodule pinned to a commit pushed to
  fusor-ai, with the browser loader and compute pool, the runtime build, the
  import check and the dev server.

What belongs where: model code (dimensions, tensor names, wiring, the image
preprocessing, the prompt and its rotary positions, the reading loop) and
anything another model could reuse go to fusor-ai, following its `AGENTS.md`.
This repository holds what turns the model into an app: the page, the checks
against the reference and the deployment. A change to fusor-ai lands there
first; this repository never patches it.

Constraints that hold everywhere:

- fusor-ai comes only from `vendor/fusor-ai`, never a crates.io version, a
  `[patch]` table or a sibling checkout. The submodule moves to another pushed
  fusor-ai commit in a change of its own, with the parity numbers and the e2e
  result it gave. No copies of fusor-ai code here.
- The reference is PaddleOCR-VL-1.6 in transformers at the version the golden
  harness pins, with its Pillow image processor and the learned position grid
  resampled with `align_corners=False`, as PaddlePaddle's own code does it (the
  checkpoint's `modeling_paddleocr_vl.py`, PaddleX and vLLM); transformers'
  port uses `True`. Preprocessing must stay bit-exact with Pillow's.
- The page owns every capability: decoding the image (file, drop or paste) to
  RGBA pixels, caching the bundle (OPFS) and measuring itself. The runtime
  receives pixels and does no I/O.
- The model is Apache-2.0: the page, the README and the Space credit PaddlePaddle
  and PaddleOCR-VL-1.6, with links.

## Checks

A change is done only when this passes:

```sh
just check
```

It runs formatting and Clippy on the page and syntax checks of the scripts;
fusor-ai's `just check` covers the runtime and the CLI. A change to the runtime,
the page, the CLI or fusor-ai also needs `just runtimes`, `just smoke`, `just
parity` (the same numbers as before unless the change is meant to move them) and
`just e2e`, whose character error rate against the reference's answers on its
fixed images must not rise; a change meant to move speed also needs `just
bench`. The model is checked stage by stage against the reference goldens by
`paddle-ocr parity`; extend it rather than adding a parallel harness.

## Measuring

A speed change is judged on the page (`web-threaded-relaxed`, 8 workers, headless
Chrome through `dump/instrumentation/trace.mjs --no-trace` or `just e2e`): the
cold and warm first token first, tokens per second second. Medians of at least
three fresh-profile runs with their spread and the machine's load average; a
change smaller than the spread is not a result, and nothing else runs on the
machine during a measurement, a kernel harness included. Traced builds (the
patches in `dump/instrumentation`) explain where the time went; accepted numbers
come from uninstrumented builds only. Numerically every accepted change is
bit-exact unless it is meant to move the numbers: the dump image reads to the
same text, `just parity` prints the same digits. fusor-ai's `knowledge.md`
records what the measurements taught about V8, the kernels and their stores.

## Commits

- Don't commit, push, tag, release or deploy unless asked. Deploying is
  described in `CONTRIBUTING.md` and is done by the maintainer.
- Git runs as `andreespirela` here: the local config pins the author, the
  credential user and the remote URL, as in the other fusor repositories. Don't
  change it, and clone fusor repositories with
  `https://andreespirela@github.com/fusor-rs/<name>.git`.
