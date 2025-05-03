//! Modul pro realizaci MAP (maximum aposteriori probability) klasifikátoru.

mod gmm;

use anyhow::Result;
use gmm::Gmm;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{array::from_fn, fs, path::Path};

/// Klasifikátor řečníka podle hlasu
#[derive(Serialize, Deserialize)]
pub struct SoundClassifier {
    classes: Vec<Class>,
}

/// Třída klasifikátoru
#[derive(Serialize, Deserialize)]
struct Class {
    name: String,
    apriori_probability: f64,
    model: Gmm,
}

impl SoundClassifier {
    /// Natrénuje model na trénovací sadě a přitom vyladí hyperparametry
    /// (počet gaussovek na třídu a počet iterací GMM) na validační sadě.
    ///
    /// Bude se snažit maximalizovat log pravděpodobnosti správné třídy.
    ///
    /// Za hyperparametry zkusí dosadit předprogramované hodnoty.
    pub fn train_tune(
        labeled_training_data: &[(String, Array2<f64>)],
        validation_data: &[(String, Array2<f64>)],
        regularization: f64,
    ) -> SoundClassifier {
        let gaussian_num_values: [usize; 20] = from_fn(|i| i + 1);
        let gmm_iters_values: [usize; 4] = from_fn(|i| 10usize.pow(i as u32 + 1));

        let mut best_found_settings = (0.0, 0, 0);
        let mut model;

        for gaussians in gaussian_num_values {
            for gmm_iters in gmm_iters_values {
                println!(
                    "Trénuju model s {:.2} gaussovkami {:.5} iterací",
                    gaussians, gmm_iters
                );
                model = SoundClassifier::train(
                    labeled_training_data,
                    gaussians,
                    gmm_iters,
                    regularization,
                );

                let nans = model.count_nans_in_parameters();
                if nans > 0 {
                    println!("Nehodnotím model, byly v něm nalezené nany");
                }

                let validation_performace: f64 = validation_data
                    .iter()
                    .map(|(class, signal)| {
                        let decision = model.classify_soft(signal.view());
                        let (_, likelihood_of_correct_class) =
                            decision.iter().find(|(cls, _)| cls == class).unwrap();
                        *likelihood_of_correct_class
                    })
                    .sum();

                if validation_performace > best_found_settings.0 {
                    best_found_settings = (validation_performace, gaussians, gmm_iters);
                }
            }
        }

        println!(
            "Nejlepší nalezené nastavení: Počet gaussovek = {}, Počet iterací = {}",
            best_found_settings.1, best_found_settings.2
        );
        println!(
            "Průměrná pravděpodobnost správné třídy = {}",
            best_found_settings.0 / validation_data.len() as f64
        );

        SoundClassifier::train(
            labeled_training_data,
            best_found_settings.1,
            best_found_settings.2,
            regularization,
        )
    }

    /// Natrénuje klasifikátor na trénovacích datech.
    /// Vezme iterátor, který iteruje `labeled_training_data` nad dvojicemi:
    /// (třída, data) a celkový počet trénovacích dat `data_len`.
    pub fn train(
        labeled_training_data: &[(String, Array2<f64>)],
        num_gaussians_for_each_class: usize,
        em_iters: usize,
        regularization: f64,
    ) -> SoundClassifier
where {
        let classes: Vec<Class> = labeled_training_data
            .into_par_iter()
            .map(|(label, data)| {
                let name = label;
                let class_len = data.dim().0;
                let apriori_probability = class_len as f64 / labeled_training_data.len() as f64;
                let model = Gmm::train(
                    data.view(),
                    num_gaussians_for_each_class,
                    em_iters,
                    regularization,
                )
                .unwrap();

                Class {
                    name: name.clone(),
                    apriori_probability,
                    model,
                }
            })
            .collect();

        SoundClassifier { classes }
    }

    /// Slouží ke klasifikaci dat pomocí modelu. Provede "soft" klasifikaci. Vrátí
    /// seznam dvojic třída - pravděpodobnost, jak moc si je model jistý, že signál patří
    /// do dané třídy.
    ///
    /// Vstupní data `signal` jsou chápána jako zvukový signál zpracovaný pomocí [`wav_to_mfcc_windows()`](crate::input::wav_to_mfcc_windows).
    /// Výstup je pole dvojic (název třídy, pravděpodobnost).
    pub fn classify_soft(&self, signal: ArrayView2<f64>) -> Vec<(&str, f64)> {
        // likelihood * apriorní pravděpodobnos každé třídy a data
        let mut probs_by_class = Array2::zeros((signal.dim().0, self.classes.len()));
        for (mut column, class) in probs_by_class
            .columns_mut()
            .into_iter()
            .zip(self.classes.iter())
        {
            let class_prob = class.model.get_prob(signal) * class.apriori_probability;
            column.assign(&class_prob);
        }

        let evidence = probs_by_class.sum_axis(Axis(1));

        // Posteriorní pravděpodobnosti každého okna signálu a třídy
        let posterior_probabilities = probs_by_class
            / &evidence
                .broadcast((self.classes.len(), signal.dim().0))
                .unwrap()
                .t();

        // Zprůměruje to pravděpodobnosti tříd přes všechna data - průměrná P třídy na celém signálu
        let overall_signal_probability = posterior_probabilities.mean_axis(Axis(0)).unwrap();

        // Přidání jména třídy
        let output_vec = Vec::from_iter(
            overall_signal_probability
                .iter()
                .zip(self.classes.iter())
                .map(|(prob, class)| (class.name.as_str(), *prob)),
        );

        output_vec
    }

    /// Stejné jako [`classify_soft()`](Self::classify_soft) až na to, že provede tvrdou klasifikaci,
    /// jednoduše vybere třídu s nejvyšší pravděpodobností a tu vrátí.
    pub fn classify_hard(&self, signal: ArrayView2<f64>) -> &str {
        let probs = self.classify_soft(signal);
        SoundClassifier::classification_hard_from_soft(probs)
    }

    /// Spočítá tvrdou klasifikaci z měkké tím, že vybere třídu s nejvyšší pravděpodobností.
    pub fn classification_hard_from_soft(soft_classification: Vec<(&str, f64)>) -> &str {
        let class_with_highest_prob = soft_classification
            .into_iter()
            .reduce(|(best_name, best_prob), (new_name, new_prob)| {
                if new_prob > best_prob {
                    (new_name, new_prob)
                } else {
                    (best_name, best_prob)
                }
            })
            .unwrap()
            .0;

        class_with_highest_prob
    }

    /// Uloží klasifikátor do souboru `file` pro pozdější načtení a použití.
    pub fn save(&self, file: &Path) -> Result<()> {
        let serialized_classifier =
            ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;

        fs::write(file, serialized_classifier)?;

        Ok(())
    }

    /// Načte klasifikátor ze souboru `file`
    pub fn load(file: &Path) -> Result<Self> {
        let serialized_classifier = fs::read_to_string(file)?;

        let classifier = ron::de::from_str(&serialized_classifier)?;

        Ok(classifier)
    }

    /// Spočítá počet hodnot NaN v parametrech modelu
    pub fn count_nans_in_parameters(&self) -> usize {
        self.classes
            .iter()
            .map(|class| class.model.count_nans_in_parameters())
            .sum()
    }
}
