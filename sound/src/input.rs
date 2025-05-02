//! Modul pro nahrávání wav souborů a jejich následné zpracování do nízkodimenzionálního
//! vektoru pro klasifikaci.

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader};
use ndarray::Array2;
use std::path::Path;

mod mfcc;

pub use mfcc::MFCCSettings;

/// Vrátí matici MFCC příznaků oken signálu
pub fn wav_to_mfcc_windows(path: &Path, mfcc_settings: &MFCCSettings) -> Result<Array2<f32>> {
    let mut reader =
        WavReader::open(path).context(format!("Nelze otevřít wav soubor {}", path.display()))?;

    if reader.spec().bits_per_sample != 16 || reader.spec().sample_format != SampleFormat::Int {
        bail!("Vzorky souboru {} nejsou ve formáty i16", path.display());
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("Vzorky souboru by měly být ve formátu i16").into())
        .collect();

    Ok(mfcc::mfcc(
        &samples,
        reader
            .spec()
            .sample_rate
            .try_into()
            .context("Nelze reprezentovat vzorkovací frekvenci")?,
        mfcc_settings,
    ))
}
