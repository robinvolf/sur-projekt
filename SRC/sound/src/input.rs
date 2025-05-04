//! Modul pro nahrávání wav souborů a jejich následné zpracování do nízkodimenzionálního
//! vektoru pro klasifikaci.

use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader};
use ndarray::{Array2, ArrayView2, Axis, concatenate};
use std::path::Path;

mod mfcc;

pub use mfcc::MFCCSettings;

use crate::classifier::SoundClassifier;

/// Vrátí matici MFCC příznaků oken signálu
pub fn wav_to_mfcc_windows(path: &Path, mfcc_settings: &MFCCSettings) -> Result<Array2<f64>> {
    let mut reader =
        WavReader::open(path).context(format!("Nelze otevřít wav soubor {}", path.display()))?;

    if reader.spec().bits_per_sample != 16 || reader.spec().sample_format != SampleFormat::Int {
        bail!("Vzorky souboru {} nejsou ve formáty i16", path.display());
    }

    let samples: Vec<f64> = reader
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

/// Načte složku `dir` s daty ve formátu:
/// ```text
/// dir
/// ├ class_dir0
/// │ ├ file.wav
/// │ └ file.wav
/// └ class_dir1
///   ├ file.wav
///   └ file.wav
/// ```
/// Pokud data nelze načíst, vrátí Error.
pub fn load_data_dir(
    dir: &Path,
    mfcc_settings: &MFCCSettings,
) -> Result<Vec<(String, Array2<f64>)>> {
    let mut labeled_data = Vec::new();

    for class_dir in dir.read_dir()? {
        let class_dir = class_dir?;

        let class_name = class_dir
            .path()
            .file_name()
            .context("Nevalidní jméno třídy")?
            .to_str()
            .context("Název třídy není validní Unicode")?
            .to_owned();

        // Kontrola, že `dir` obsahuje pouze složky
        if !class_dir.path().is_dir() {
            bail!("Ve složce trénovacích složek je soubor! {}", class_name);
        }

        let class_iter = class_dir.path().read_dir()?;

        let mut samples = Vec::new();
        for file in class_iter {
            let file = file.context(format!("Nelze přečíst soubor v {class_name}"))?;
            let mfcc_samples =
                wav_to_mfcc_windows(&file.path(), mfcc_settings).context(format!(
                    "Nelze zpracovat soubor {} na MFCC příznaky",
                    file.path().display(),
                ))?;
            samples.push(mfcc_samples);
        }
        let samples_views: Vec<ArrayView2<f64>> = samples.iter().map(|s| s.view()).collect();

        // Všechny okýnka jedné třídy poskládané ze všech nahrávek
        let class_samples = concatenate(Axis(0), samples_views.as_slice()).context(format!(
            "Nelze spojit všechna trénovací data třídy {}",
            class_name
        ))?;

        labeled_data.push((class_name, class_samples));
    }

    Ok(labeled_data)
}

/// Převede výsledek měkké klasifikace na řetězec ve formátu spefikovaném v zadání:
///
/// ### Formát
/// N + 2 polí na řádku oddělené mezerou. (N = počet tříd)
/// Tyto pole budou obsahovat popořadě následující údaje:
///
///  - jméno segmentu (jméno souboru BEZ přípony .wav či .png)
///  - tvrdé rozhodnutí o třídě
///  - následujících N polí bude popořadě obsahovat číselná skóre odpovídající
///    logaritmickým pravděpodobnostem jednotlivých tříd 1 až N.
///    (Pokud použijete klasifikátor jehož výstup se nedá interpretovat
///    pravděpodobnostně, nastavte tato pole na hodnotu NaN.
pub fn classification_format(file_name: &Path, decision: Vec<(&str, f64)>) -> String {
    let mut output = String::new();
    let segment_name = file_name.file_stem().unwrap().to_string_lossy();
    let hard_decision = SoundClassifier::classification_hard_from_soft(decision.clone());

    output.push_str(&segment_name);
    output.push(' ');
    output.push_str(&hard_decision);

    for (_, prob) in decision {
        output.push(' ');
        output.push_str(&format!("{:.4}", prob.ln()));
    }

    output
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn classification_format_test() {
        let filename = PathBuf::from("some_audio_file.wav");
        let decision = vec![("0", 0.1), ("1", 0.3), ("2", 0.5), ("3", 0.1)];

        let format = classification_format(&filename, decision);

        assert_eq!(
            format,
            "some_audio_file 2 -2.3026 -1.2040 -0.6931 -2.3026\n"
        )
    }
}
