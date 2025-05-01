//! Modul pro realizaci MAP (maximum aposteriori probability) klasifikátoru.

mod gmm;

use gmm::{DEFAULT_NUM_CLUSTERS, Gmm};
use ndarray::{Array2, ArrayView2, Axis};

/// Klasifikátor řečníka podle hlasu
struct SoundClassifier {
    classes: Vec<Class>,
}

/// Třída klasifikátoru
struct Class {
    name: String,
    apriori_probability: f32,
    model: Gmm,
}

impl SoundClassifier {
    /// Natrénuje klasifikátor na trénovacích datech.
    /// Vezme iterátor, který iteruje `labeled_training_data` nad dvojicemi:
    /// (třída, data) a celkový počet trénovacích dat `data_len`.
    pub fn train<'tr, I>(labeled_training_data: I, data_len: usize) -> SoundClassifier
    where
        I: IntoIterator<Item = (String, ArrayView2<'tr, f32>)>,
    {
        let classes: Vec<Class> = labeled_training_data
            .into_iter()
            .map(|(label, data)| {
                let name = label;
                let apriori_probability = data.dim().0 as f32 / data_len as f32;
                let model = Gmm::train(data, DEFAULT_NUM_CLUSTERS).unwrap();

                Class {
                    name,
                    apriori_probability,
                    model,
                }
            })
            .collect();

        SoundClassifier { classes }
    }

    /// Slouží ke klasifikaci dat pomocí modelu
    /// Vstupní data `signal` jsou chápána jako zvukový signál zpracovaný pomocí `[crate::mfcc:mfcc]`.
    /// Výstup je pole dvojic (název třídy, pravděpodobnost).
    ///
    /// Pozn. pokud vybíráme třídu s maximální posteriorní pravděpodobností,
    /// můžeme vynechat `p(x)`, je u všech tříd stejné.
    pub fn classify(&self, signal: ArrayView2<f32>) -> Vec<(&str, f32)> {
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
                .broadcast((signal.dim().0, self.classes.len()))
                .unwrap();

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
}
