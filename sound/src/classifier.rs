//! Modul pro realizaci MAP (maximum aposteriori probability) klasifikátoru.

mod gmm;

use gmm::{DEFAULT_NUM_CLUSTERS, Gmm};
use ndarray::{Array2, ArrayView1, ArrayView2};

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

    /// Klasifikace založená na maximální posteriorní pravděpodobnosti:
    /// `P(C|x) = (p(x|C) * P(C))/p(x)`
    ///
    /// Pozn. pokud vybíráme třídu s maximální posteriorní pravděpodobností,
    /// můžeme vynechat `p(x)`, je u všech tříd stejné.
    pub fn classify(&self, data: ArrayView2<f32>) -> &str {
        self.classes.iter().map(|class| {
            let likelihood = class.model.get_prob(data);
            let likelihod_times_apriori = likelihood * class.apriori_probability;

            todo!()
        });

        todo!();
    }
}
