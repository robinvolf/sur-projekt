//! Modul pro realizaci MAP (maximum aposteriori probability) klasifikátoru.

mod gmm;

use std::{fs, path::Path};

use anyhow::Result;
use gmm::Gmm;
use ndarray::{Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

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
    /// Natrénuje klasifikátor na trénovacích datech.
    /// Vezme iterátor, který iteruje `labeled_training_data` nad dvojicemi:
    /// (třída, data) a celkový počet trénovacích dat `data_len`.
    pub fn train<'tr, I>(
        labeled_training_data: I,
        num_gaussians_for_each_class: usize,
        em_iters: usize,
        regularization: f64,
    ) -> SoundClassifier
    where
        I: IntoIterator<Item = (String, ArrayView2<'tr, f64>)>,
    {
        let mut data_len = 0.0;
        let mut classes: Vec<Class> = labeled_training_data
            .into_iter()
            .map(|(label, data)| {
                let name = label;
                let apriori_probability = data.dim().0 as f64; // Zatím to není apriorní pravděpodobnost, je to jen počet dat v dané třídě
                let model =
                    Gmm::train(data, num_gaussians_for_each_class, em_iters, regularization)
                        .unwrap();

                data_len += apriori_probability;

                Class {
                    name,
                    apriori_probability,
                    model,
                }
            })
            .collect();

        // Podělení délkou dat, abychom počet prvků třídy změnili na apriorní pravděpodobnost
        classes
            .iter_mut()
            .for_each(|class| class.apriori_probability /= data_len);

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
