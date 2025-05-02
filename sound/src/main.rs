use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use classifier::SoundClassifier;
use input::{MFCCSettings, wav_to_mfcc_windows};
use ndarray::{Array2, ArrayView2, Axis, concatenate};
use std::path::{Path, PathBuf};

mod classifier;
mod input;

#[derive(Parser)]
struct Config {
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

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Natrénuje klasifikátor na dodaných datech.
    Train {
        /// Složka se složkami, kde jsou uložené testovací vzory. Každá složka uvnitř `dir`
        /// je považována za jednu třídu, jejíž název odpovídá názvu složky.
        dir: PathBuf,
        /// Soubor, do kterého se posléze uloží klasifikátor pro pozdější použití.
        save_file: PathBuf,
        /// Počet gaussovek pro každou třídu v GMM
        #[arg(long, default_value_t = 5)]
        gaussians: usize,
    },
    /// Použije klasifikátor ke klasifikaci dodaných dat
    Classify {
        /// Soubor, ze kterého se má klasifikátor přečíst
        load_file: PathBuf,
        /// Jednotlivé .wav soubory ke klasifikaci
        recordings: Vec<PathBuf>,
    },
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

fn load_training_dir(
    dir: &Path,
    mfcc_settings: &MFCCSettings,
) -> Result<Vec<(String, Array2<f32>)>> {
    let mut labeled_data = Vec::new();

    for class_dir in dir.read_dir()? {
        let class_dir = class_dir?;

        let class_name = class_dir.path().display().to_string();

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
        let samples_views: Vec<ArrayView2<f32>> = samples.iter().map(|s| s.view()).collect();

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
/// 33 polí na řádku oddělené mezerou.
/// Tyto pole budou obsahovat popořadě následující údaje:
///
///  - jméno segmentu (jméno souboru BEZ přípony .wav či .png)
///  - tvrdé rozhodnutí o třídě, kterým bude celé číslo s hodnotou od 1 do 31.
///  - následujících 31 polí bude popořadě obsahovat číselná skóre odpovídající
///    logaritmickým pravděpodobnostem jednotlivých tříd 1 až 31.
///    (Pokud použijete klasifikátor jehož výstup se nedá interpretovat
///    pravděpodobnostně, nastavte tato pole na hodnotu NaN.
fn classification_format(file_name: &Path, decision: Vec<(&str, f32)>) -> String {
    todo!()
}

fn main() -> Result<()> {
    let config = Config::parse();
    let mfcc_settings = MFCCSettings::from(&config);

    match config.command {
        Command::Train {
            dir,
            save_file,
            gaussians,
        } => {
            println!("Načítám trénovací data");
            let training_data = load_training_dir(&dir, &mfcc_settings)
                .context("Nepodařilo se načíst trénovací data")?;

            println!("Trénuji model...");
            let model = SoundClassifier::train(
                training_data
                    .iter()
                    .map(|(str, samp)| (str.clone(), samp.view())),
                gaussians,
            );

            println!("Model natrénován, ukládám do {}", save_file.display());
            model
                .save(&save_file)
                .context("Nepodařilo se uložit model")?;

            Ok(())
        }
        Command::Classify {
            load_file,
            recordings,
        } => {
            let model = SoundClassifier::load(&load_file).context("Nepodařilo se načíst model")?;
            for file in recordings {
                let file_mfcc = wav_to_mfcc_windows(&file, &mfcc_settings).context(format!(
                    "Nelze zpracovat soubor {} na MFCC příznaky",
                    file.display()
                ))?;

                let soft_decision = model.classify_soft(file_mfcc.view());
                let formatted_result = classification_format(&file, soft_decision);
                println!("{}", formatted_result);
            }

            Ok(())
        }
    }
}
