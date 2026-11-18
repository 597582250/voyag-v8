//! CLI demo for the Rust binding.
//!
//! Usage: `voyag-cli <model-dir> <utterance>`

use std::env;
use std::process;

use voyag::VoyagPredictor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: voyag-cli <model-dir> <utterance>");
        process::exit(2);
    }
    let predictor = VoyagPredictor::create(&args[1])?;
    let result = predictor.predict(&args[2], None)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
