use anyhow::Result;
use clap::Parser;
use input::MFCCSettings;
use std::path::PathBuf;

mod classifier;
mod input;

#[derive(Parser)]
struct Config {
    /// Cesta k .wav souboru
    #[arg(short, long)]
    wav_path: PathBuf,

    /// Velikost okna v ms
    #[arg(long, default_value_t = 100)]
    window_size_ms: u32,

    /// Počet vzorků, o které se budou okna překrývat
    #[arg(long, default_value_t = 400)]
    windows_overlap: usize,

    /// Počet Mel-filter bank, které budou aplikovány na signál
    #[arg(long, default_value_t = 100)]
    num_mel_filter_banks: usize,

    /// Maximální počet MFCC koeficientů, které budou vybrány
    #[arg(long, default_value_t = 20)]
    atmost_coeffs: usize,
}

impl From<&Config> for MFCCSettings {
    fn from(value: &Config) -> Self {
        Self {
            window_size_ms: value.window_size_ms,
            windows_overlap: value.windows_overlap,
            num_mel_filter_banks: value.num_mel_filter_banks,
            atmost_coeffs: value.atmost_coeffs,
        }
    }
}

fn main() -> Result<()> {
    let config = Config::parse();

    let mfcc_setting = MFCCSettings::from(&config);
    let mfcc = input::wav_to_mfcc_windows(&config.wav_path, &mfcc_setting)?;

    println!("{:?}", mfcc.shape());
    println!("{}", mfcc);

    Ok(())
}
