use anyhow::{Result, anyhow};
use clap::Parser;
use plotters::prelude::*;
use std::path::{Path, PathBuf};

mod input;

#[derive(Parser)]
struct Config {
    #[arg(short)]
    wav_path: PathBuf,
}

fn main() -> Result<()> {
    let config = Config::parse();

    input::wav_to_mfcc(&config.wav_path);

    Ok(())
}
