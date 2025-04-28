//! Modul pro nahrávání wav souborů a jejich následné zpracování do nízkodimenzionálního
//! vektoru pro klasifikaci.

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader};
use mfcc::mfcc;
use ndarray::Array2;
use std::path::Path;

mod mfcc;

/// Velikost okna v ms
const WINDOW_SIZE_MS: u32 = 100;

/// Počet vzorků, o které se budou okna překrývat
const WINDOWS_OVERLAP: usize = 400;

/// Počet Mel-filter bank, které budou aplikovány na signál
const NUM_MEL_FILTER_BANKS: usize = 100;

/// Maximální počet MFCC koeficientů, které budou vybrány
const MAX_MFCC_COEFFS: usize = 20;

pub fn wav_to_mfcc(path: &Path) -> Result<Array2<f32>> {
    let mut reader =
        WavReader::open(path).context(format!("Nelze otevřít wav soubor {}", path.display()))?;

    if reader.spec().bits_per_sample != 16 || reader.spec().sample_format != SampleFormat::Int {
        bail!("Vzorky souboru {} nejsou ve formáty i16", path.display());
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("Vzorky souboru by měly být ve formátu i16").into())
        .collect();

    let samples_per_window =
        mfcc::samples_in_window(reader.spec().sample_rate as f32, WINDOW_SIZE_MS);

    Ok(mfcc(
        &samples,
        reader.spec().sample_rate as usize,
        WINDOWS_OVERLAP,
        samples_per_window,
        NUM_MEL_FILTER_BANKS,
        MAX_MFCC_COEFFS,
    ))
}
