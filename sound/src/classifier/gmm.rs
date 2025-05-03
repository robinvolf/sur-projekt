//! Modul pro práci s GMM (Gaussian mixture model).

use std::{f64::consts::PI, iter::repeat_n};

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::{Determinant, Inverse};
use ndarray_stats::CorrelationExt;
use rand::Rng;
use rand_distr::{StandardNormal, num_traits::Inv};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct GmmGaussian {
    /// Pravděpodobnost výběru dané gaussovky v GMM
    prob: f64,
    /// Střední hodnota
    mean: Array1<f64>,
    /// Kovarianční matice
    covariance_matrix: Array2<f64>,
}

impl GmmGaussian {
    /// Spočítá pravděpodobnost, že data pocházejí z této gaussovky v rámci GMM.
    fn get_prob(&self, data: ArrayView2<f64>) -> Array1<f64> {
        // Při výpočtu pravděpodobnosti je tato část výpočtu vždy stejná
        let inv_cov_matrix = self
            .covariance_matrix
            .inv()
            .expect("Nelze spočítat inverzní kovarianční matici");
        let cov_matrix_det = self
            .covariance_matrix
            .det()
            .expect("Nelze spočítat determinant kovarianční matice");
        let dimensionality = data.dim().1 as i32;

        // Pole pravděpodobností z gaussovky
        let mut probs = Array1::from_iter(data.outer_iter().map(|features| {
            let exponent = -0.5
                * (&features - &self.mean)
                    .dot(&inv_cov_matrix)
                    .dot(&(&features - &self.mean));

            ((2.0 * PI).powi(dimensionality) * cov_matrix_det)
                .sqrt()
                .inv()
                * exponent.exp()
        }));

        // Normalizace pravděpodobností výběru této gaussovky v rámci GMM
        probs *= self.prob;

        probs
    }
}

/// Struktura reprezentující generativní model, který modeluje data
/// pomocí směsice Gaussovských rozložení. Data jsou N-dimenzionální vektory.
#[derive(Serialize, Deserialize, Debug)]
pub struct Gmm {
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
    pub fn train(
        training_data: ArrayView2<f64>,
        num_gaussians: usize,
        em_iters: usize,
        regularization: f64,
    ) -> Result<Self> {
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
            mean.mapv_inplace(|x| x + 0.001 * rng.sample::<f64, _>(StandardNormal));
            covariance_matrix.mapv_inplace(|x| x + 0.001 * rng.sample::<f64, _>(StandardNormal));
        }

        for _ in 0..em_iters {
            // Expectation
            let responsibilities = gmm.calculate_responsibilities(training_data);

            // Maximization
            gmm.update_params(responsibilities.view(), training_data, regularization);
        }

        Ok(gmm)
    }

    /// Vrátí podmíněnou pravděpodobnost `P(x | params)`, tedy pravděpodobnost, že tento model
    /// vygeneroval daná data při aktuálním nastavení parametrů.
    pub fn get_prob(&self, data: ArrayView2<f64>) -> Array1<f64> {
        let mut result = Array2::zeros((data.dim().0, self.gaussians.len()));

        for (mut column, gaussian) in result.columns_mut().into_iter().zip(self.gaussians.iter()) {
            column.assign(&gaussian.get_prob(data));
        }

        result.sum_axis(Axis(1))
    }

    /// Spočítá pro model "responsibilities" - tj. jak moc jsou jednotlivé gaussovky
    /// zodpovědné za vysvětlení výskytu každého data.
    ///
    /// Matice je ve tvaru: (C = počet gaussovek, D = počet dat)
    /// ```
    ///    C
    /// [ ... ]
    /// [ ... ] D
    /// [ ... ]
    /// ```
    fn calculate_responsibilities(&self, training_data: ArrayView2<f64>) -> Array2<f64> {
        let data_len = training_data.dim().0;

        // Pro každé dato a gaussovku spočítám pravděpodobnost, že daná gaussovka vygenerovala dané dato a zváhuju to pravděopdobností, výběru dané gaussovky.
        let mut weighted_probs_from_gaussians = Array2::zeros((data_len, self.gaussians.len()));
        for (gaussian, mut column) in self
            .gaussians
            .iter()
            .zip(weighted_probs_from_gaussians.columns_mut())
        {
            let probabilites_from_gaussian = gaussian.get_prob(training_data);
            column.assign(&probabilites_from_gaussian);
        }

        let resp_denom = weighted_probs_from_gaussians.sum_axis(Axis(1));

        let responsibilities = weighted_probs_from_gaussians
            / resp_denom
                .broadcast((self.gaussians.len(), data_len))
                .unwrap()
                .t();

        responsibilities
    }

    fn update_params(
        &mut self,
        responsibilities: ArrayView2<f64>,
        training_data: ArrayView2<f64>,
        regularization: f64,
    ) {
        let responsibilities_sum = responsibilities.sum();

        for (
            GmmGaussian {
                prob,
                mean,
                covariance_matrix,
            },
            responsibilities,
        ) in self.gaussians.iter_mut().zip(responsibilities.columns())
        {
            let new_mean = responsibilities.sum().inv()
                * (&training_data
                    * &responsibilities
                        .broadcast((self.dimensionality, training_data.dim().0))
                        .unwrap()
                        .t())
                    .sum_axis(Axis(0));
            mean.assign(&new_mean);

            let mut new_cov = Array2::eye(self.dimensionality) * regularization;
            for (features, responsibility) in training_data
                .rows()
                .into_iter()
                .zip(responsibilities.iter())
            {
                let left = (&features - &new_mean)
                    .broadcast((self.dimensionality, self.dimensionality))
                    .unwrap()
                    .t()
                    .to_owned();
                let right = (&features - &new_mean)
                    .broadcast((self.dimensionality, self.dimensionality))
                    .unwrap()
                    .to_owned();
                new_cov += &(*responsibility * left * right);
            }
            covariance_matrix.assign(&new_cov);
            covariance_matrix.mapv_inplace(|x| x / responsibilities.sum());

            *prob = responsibilities.sum() / responsibilities_sum;
        }
    }

    /// Inicializuje GMM tak, že každé gaussovce přiřadí stejný průměr
    /// a stejnou kovarianční matici.
    fn initialize_same_from_data(
        training_data: ArrayView2<f64>,
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

        let overall_prob = 1.0 / num_gaussians as f64;

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

    /// Spočítá počet hodnot v Nan v parametrech
    pub fn count_nans_in_parameters(&self) -> usize {
        self.gaussians
            .iter()
            .map(|gaussian| {
                usize::from(gaussian.prob.is_nan())
                    + gaussian
                        .mean
                        .iter()
                        .fold(0, |nans, x| nans + usize::from(x.is_nan()))
                    + gaussian
                        .covariance_matrix
                        .iter()
                        .fold(0, |nans, x| nans + usize::from(x.is_nan()))
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use ndarray_linalg::assert_aclose;

    use super::*;

    const PRECISION: f64 = 0.001;

    #[test]
    fn test_gmm_initialization() {
        let training_data = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let gmm = Gmm::initialize_same_from_data(training_data.view(), 2)
            .expect("Výpočet by měl proběhout v pořádku");

        assert_eq!(gmm.gaussians.len(), 2);
        for gaussian in gmm.gaussians {
            assert_eq!(gaussian.prob, 0.5);
            assert_eq!(gaussian.mean, array![2.0, 2.0]);
            assert_eq!(gaussian.covariance_matrix, array![[1.0, 1.0], [1.0, 1.0]]);
        }
    }

    #[test]
    fn train_test() {
        let training_data = include!("gmm_test_data.rs");

        let gmm = Gmm::train(training_data.view(), 3, 100, 0.0).unwrap();
        let prob = 1.0 / 3.0; // Všechny gussovky by měly mít stejný počet hodnot
        let limit = PRECISION;

        let expected_gaussians = vec![
            GmmGaussian {
                prob,
                mean: array![50.0, 40.0],
                covariance_matrix: array![[100.0, 70.0], [70.0, 100.0]],
            },
            GmmGaussian {
                prob,
                mean: array![40.0, 75.0],
                covariance_matrix: array![[25.0, 0.0], [0.0, 25.0]],
            },
            GmmGaussian {
                prob,
                mean: array![10.0, 60.0],
                covariance_matrix: array![[5.0, 0.0], [0.0, 100.0]],
            },
        ];

        println!("{:#?}", gmm);

        assert_eq!(gmm.count_nans_in_parameters(), 0);
        for (gaussian, expected) in gmm
            .gaussians
            .into_iter()
            .zip(expected_gaussians.into_iter())
        {
            assert_aclose!(gaussian.prob, expected.prob, limit);
            for (got, expected) in gaussian.mean.iter().zip(expected.mean.iter()) {
                assert_aclose!(*got, *expected, limit);
            }
            for (got, expected) in gaussian
                .covariance_matrix
                .iter()
                .zip(expected.covariance_matrix.iter())
            {
                assert_aclose!(*got, *expected, limit);
            }
        }
    }

    #[test]
    fn get_prob_gaussian_test() {
        let gaussian = GmmGaussian {
            prob: 1.0,
            mean: array![50.0, 40.0],
            covariance_matrix: array![[100.0, 70.0], [70.0, 100.0]],
        };
        let expected_prob = 0.0022286149708619224;
        let prob = gaussian.get_prob(array![[50.0, 40.0]].view());

        assert_aclose!(prob[0], expected_prob, PRECISION);
    }
}
