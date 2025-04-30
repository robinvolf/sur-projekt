//! Modul pro práci s GMM (Gaussian mixture model).

use std::{
    f32::consts::PI,
    iter::{Map, repeat_n},
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{
    Array0, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, IntoNdProducer, ShapeBuilder, s,
};
use ndarray_linalg::{Determinant, Inverse};
use ndarray_stats::{CorrelationExt, SummaryStatisticsExt};
use rand::{Rng, seq::IndexedRandom};
use rand_distr::{Distribution, Normal, StandardNormal, num_traits::Inv};

#[derive(Clone)]
struct GmmGaussian {
    /// Pravděpodobnost výběru dané gaussovky v GMM
    prob: f32,
    /// Střední hodnota
    mean: Array1<f32>,
    /// Kovarianční matice
    covariance_matrix: Array2<f32>,
}

impl GmmGaussian {
    /// Spočítá pravděpodobnost, že data pocházejí z této gaussovky v rámci GMM.
    fn get_prob(&self, data: ArrayView2<f32>) -> Array1<f32> {
        // Při výpočtu pravděpodobnosti je tato část výpočtu vždy stejná
        let multiplier = (2.0 * PI).powf(-(data.dim().1 as f32) / 2.0);
        let inv_cov_matrix = self
            .covariance_matrix
            .inv()
            .expect("Nelze spočítat inverzní kovarianční matici");
        let cov_matrix_det = self
            .covariance_matrix
            .det()
            .expect("Nelze spočítat determinant kovarianční matice");

        // Pole pravděpodobností z gaussovky
        let mut probs = Array1::from_iter(data.outer_iter().map(|features| {
            let exponent = (&features - &self.mean)
                .dot(&inv_cov_matrix)
                .dot(&(&features - &self.mean));

            multiplier * cov_matrix_det.inv().sqrt() * exponent.exp()
        }));

        // Normalizace pravděpodobností výběru této gaussovky v rámci GMM
        probs *= self.prob;

        probs
    }
}

/// Struktura reprezentující generativní model, který modeluje data
/// pomocí směsice Gaussovských rozložení. Data jsou N-dimenzionální vektory.
struct Gmm {
    dimensionality: usize,
    gaussians: Vec<GmmGaussian>,
}

impl Gmm {
    /// Vytvoří nový GMM model z natrénovaných dat `training_data`,
    /// který modeluje data pomocí `num_gaussians` gaussovek.
    ///
    /// ### Trénování
    /// Pro trénování používá algoritmus
    /// [Expectation-Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).
    pub fn train(training_data: ArrayView2<f32>, num_gaussians: usize) -> Result<Self> {
        // Nejdříve inicializujeme parametry všech gaussovek na stejnou hodnotu spočítané z dat
        let mut gmm = Gmm::initialize_same_from_data(training_data, num_gaussians)?;

        // Přidáme trochu šumu, aby byly jednotlivé gaussovky odlišné
        let mut rng = rand::rng();
        for GmmGaussian {
            mean,
            covariance_matrix,
            ..
        } in gmm.gaussians.iter_mut()
        {
            mean.mapv_inplace(|x| x + 0.001 * rng.sample::<f32, _>(StandardNormal));
            covariance_matrix.mapv_inplace(|x| x + 0.001 * rng.sample::<f32, _>(StandardNormal));
        }

        let should_terminate = false;
        while !should_terminate {
            // Expectation

            let responsibilities = gmm.calculate_responsibilities(training_data);

            debug_assert_eq!(
                responsibilities.sum_axis(Axis(1)),
                Array1::from_elem(responsibilities.dim().0, 1.0),
                "Suma přes pravděpodobnosti pro každé dato by měla být 1.0"
            );

            // Maximization
        }

        todo!()
    }

    fn calculate_responsibilities(&self, training_data: ArrayView2<f32>) -> Array2<f32> {
        // Pro každé dato a gaussovku spočítám pravděpodobnost, že daná gaussovka vygenerovala dané dato a zváhuju to pravděopdobností, výběru dané gaussovky
        let mut weighted_probs_from_gaussians = Array2::zeros((0, self.gaussians.len()));
        for gaussian in self.gaussians.iter() {
            let probabilites_from_gaussian = gaussian.get_prob(training_data);
            weighted_probs_from_gaussians
                .push_row(probabilites_from_gaussian.view())
                .unwrap();
        }

        let resp_denom = weighted_probs_from_gaussians.sum_axis(Axis(1));

        // Pro každé dato to říká, jak moc jsou jednotlivé gaussovky za to dato zodpovědné
        let responsibilities = weighted_probs_from_gaussians / &resp_denom.t(); // Musíme transponovat, v responsibilities jsou data po řádcích a gaussovky po sloupcích

        responsibilities
    }

    //
    fn update_params(&mut self, responsibilities: ArrayView2<f32>, training_data: ArrayView2<f32>) {
        // Váhy jednotlivých gaussovek, když vezmeme v úvahu všechna data (kolik proporčně dat náleží každé z gaussovek)
        let gauss_responsibilities = responsibilities.sum_axis(Axis(0));

        // Spočítáme nové pravděpodobnosti jednotlivých gaussovek
        let new_gauss_probs = gauss_responsibilities / responsibilities.sum();

        // Spočítáme nové střední hodnoty a kovarianční matice
        // let weighted_data = &training_data
        //     .t()
        //     .broadcast((dimensionality, training_data_len, num_gaussians))
        //     .expect("Nelze rozšířit trénovací data")
        //     * &responsibilities
        //         .broadcast((dimensionality, training_data_len, num_gaussians))
        //         .expect("Nelze rozšířit responsibilities");
        // let new_means = weighted_data.sum_axis(Axis(1))
        //     / new_gauss_probs
        //         .broadcast((dimensionality, num_gaussians))
        //         .unwrap();

        // let new_covs = todo!();
    }

    /// Inicializuje GMM tak, že každé gaussovce přiřadí stejný průměr
    /// a stejnou kovarianční matici.
    fn initialize_same_from_data(
        training_data: ArrayView2<f32>,
        num_gaussians: usize,
    ) -> Result<Self> {
        let dimensionality = training_data.dim().1;

        let overall_mean = training_data
            .mean_axis(Axis(0))
            .context("Nelze spočítat celkový průměr")?;
        // Musíme transponovat, kovariance se počítá nad jednotlivými řádky, ale my máme jednotlivé proměnné po sloupcích.
        // 1.0 Protože počítáme kovarianční matici nad vzorky, ne nad populací
        let overall_cov_matrix = training_data
            .t()
            .cov(1.0)
            .context("Nelze spočítat celkovou kovarianční matici")?;

        let overall_prob = 1.0 / num_gaussians as f32;

        let gaussians = repeat_n(
            GmmGaussian {
                prob: overall_prob,
                mean: overall_mean,
                covariance_matrix: overall_cov_matrix,
            },
            num_gaussians,
        )
        .collect();

        Ok(Gmm {
            dimensionality,
            gaussians,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_gmm_initialization() {
        let training_data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let gmm = Gmm::initialize_same_from_data(training_data.view(), 2)
            .expect("Výpočet by měl proběhout v pořádku");

        assert_eq!(gmm.gaussians.len(), 2);
        for gaussian in gmm.gaussians {
            assert_eq!(gaussian.mean, array![2.0, 2.0]);
            assert_eq!(gaussian.covariance_matrix, array![[1.0, 1.0], [1.0, 1.0]]);
        }
    }
}
