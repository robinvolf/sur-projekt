use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod input;

#[derive(Parser)]
struct Config {
    #[arg(short)]
    wav_path: PathBuf,
}

fn main() -> Result<()> {
    let config = Config::parse();

    let mfcc = input::wav_to_mfcc(&config.wav_path)?;

    println!("{}", mfcc);

    Ok(())
}
