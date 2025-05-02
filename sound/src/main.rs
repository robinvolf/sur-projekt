use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use classifier::SoundClassifier;
use input::{MFCCSettings, classification_format, load_training_dir, wav_to_mfcc_windows};
use std::path::PathBuf;

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
        /// Počet iterací EM algoritmu při trénování GMM
        #[arg(long, default_value_t = 10)]
        em_iters: usize,
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

fn main() -> Result<()> {
    let config = Config::parse();
    let mfcc_settings = MFCCSettings::from(&config);

    match config.command {
        Command::Train {
            dir,
            save_file,
            gaussians,
            em_iters,
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
                em_iters,
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
