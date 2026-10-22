//! `paddle-ocr bench`: the milliseconds of each stage on one image, and of reading it whole.

use std::time::Instant;

use anyhow::Result;
use paddle_ocr::pipeline::Stage;
use paddle_ocr::prompt::Task;
use paddle_ocr::session::{Engine, Request};
use paddle_ocr::{Patches, Pixels};

const CALLS: usize = 3;
/// Answer tokens the decode stage is timed over.
const DECODE_TOKENS: usize = 32;

/// Milliseconds per call of `call`, after one warm-up call.
fn time<T>(mut call: impl FnMut() -> Result<T, paddle_ocr::Error>) -> Result<f64> {
    call()?;
    let start = Instant::now();
    for _ in 0..CALLS {
        call()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / CALLS as f64)
}

pub fn run(engine: &mut Engine, image: &Pixels, task: Task) -> Result<()> {
    let patches = Patches::of(image)?;
    let grid = patches.grid;
    let prompt = engine.prompt(task, grid)?;
    println!(
        "threads {}, {}×{} image, {}×{} patches, {} prompt tokens",
        engine.threads(),
        image.size.width,
        image.size.height,
        grid.rows,
        grid.columns,
        prompt.ids.len()
    );
    let embedded = engine.embed(&patches);
    let seen = engine.encode(&embedded, grid);
    let tokens = engine.project(&seen, grid);
    for stage in Stage::ALL {
        let ms = match stage {
            Stage::Preprocess => time(|| Patches::of(image))?,
            Stage::Embed => time(|| Ok(engine.embed(&patches)))?,
            Stage::Encode => time(|| Ok(engine.encode(&embedded, grid)))?,
            Stage::Project => time(|| Ok(engine.project(&seen, grid)))?,
            Stage::Prefill => time(|| Ok(engine.prefill(&prompt, &tokens)))?,
            Stage::Decode => {
                let answer = |engine: &mut Engine| engine.answer(&prompt, &tokens, DECODE_TOKENS);
                let written = answer(engine).len();
                let with_prefill = time(|| Ok(answer(engine)))?;
                let prefill = time(|| Ok(engine.prefill(&prompt, &tokens)))?;
                (with_prefill - prefill) / written.saturating_sub(1).max(1) as f64
            }
        };
        println!("  {:12} {ms:9.1} ms", stage.name());
    }
    let request = Request {
        task,
        max_tokens: None,
    };
    let start = Instant::now();
    engine.start(image, &request)?;
    let mut progress = engine.step()?;
    while !progress.done {
        progress = engine.step()?;
    }
    println!(
        "  read whole   {:9.1} ms, {} tokens",
        start.elapsed().as_secs_f64() * 1000.0,
        progress.tokens
    );
    Ok(())
}
