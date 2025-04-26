//! Modul pro nahrávání wav souborů a jejich následné zpracování do nízkodimenzionálního
//! vektoru pro klasifikaci.

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader};
use ndarray::Array2;
use std::path::Path;

mod mfcc;

use mfcc::{samples_in_window, samples_into_window_matrix};

pub fn wav_to_mfcc(path: &Path) -> Result<Array2<i16>> {
    todo!()
}

/// Načte wav soubor `path` ve formátu 16bit a vytvoří matici oken přes tento soubor,
/// kde každé okno bude `window_size_ms` dlouhé. Pokud je problém z načítáním,
/// vrátí Error.
///
/// ### Tvar matice
/// Vrácená matice je tohoto tvaru, kde:
///   - `K` = Počet vzorků v jednom okně
///   - `N` = Počet jednotlivých oken
/// ```
///    K
/// [ ... ]
/// [ ... ]
/// [ ... ] N
/// [ ... ]
/// [ ... ]
/// ```
fn load_wav_file_16bit(
    path: &Path,
    window_size_ms: u32,
    windows_overlap: u32,
) -> Result<Array2<i16>> {
    let mut reader =
        WavReader::open(path).context(format!("Nelze otevřít wav soubor {}", path.display()))?;

    if reader.spec().bits_per_sample != 16 || reader.spec().sample_format != SampleFormat::Int {
        bail!("Vzorky souboru {} nejsou ve formáty i16", path.display());
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("Vzorky souboru by měly být ve formátu i16"))
        .collect();

    let samples_per_window = samples_in_window(reader.spec().sample_rate as f32, window_size_ms);
    let matrix = samples_into_window_matrix(&samples, samples_per_window, windows_overlap as usize);

    Ok(matrix)
}
