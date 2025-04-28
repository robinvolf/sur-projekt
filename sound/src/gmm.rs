//! Modul pro práci s GMM (Gaussian mixture model).

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use ndarray_stats::{CorrelationExt, SummaryStatisticsExt};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Struktura reprezentující generativní model, který modeluje data
/// pomocí směsice Gaussovských rozložení. Data jsou N-dimenzionální vektory.
struct Gmm {
    /// Počet gaussovek, které model používá pro modelování dat
    num_gaussians: usize,
    /// Střední hodnoty jednotlivých gaussovek
    means: Array2<f32>,
    /// Kovarianční matice jednotlivých gaussovek
    covariance_matrices: Array3<f32>,
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
        gmm.means
            .mapv_inplace(|x| x + 0.001 * rng.sample::<f32, _>(StandardNormal));
        gmm.covariance_matrices
            .mapv_inplace(|x| x + 0.001 * rng.sample::<f32, _>(StandardNormal));

        todo!()
    }

    /// Inicializuje GMM tak, že každé gaussovce přiřadí stejný průměr
    /// a stejnou kovarianční matici.
    fn initialize_same_from_data(
        training_data: ArrayView2<f32>,
        num_gaussians: usize,
    ) -> Result<Self> {
        let overall_mean = training_data
            .mean_axis(Axis(0))
            .context("Nelze spočítat celkový průměr")?;
        // Musíme transponovat, kovariance se počítá nad jednotlivými řádky, ale my máme jednotlivé proměnné po sloupcích.
        // 1.0 Protože počítáme kovarianční matici nad vzorky, ne nad populací
        let overall_cov_matrix = training_data
            .t()
            .cov(1.0)
            .context("Nelze spočítat celkovou kovarianční matici")?;

        let mut means = Array2::zeros((num_gaussians, overall_mean.dim()));
        means.assign(&overall_mean);

        let mut covariance_matrices = Array3::zeros((
            num_gaussians,
            overall_cov_matrix.dim().0,
            overall_cov_matrix.dim().1,
        ));
        covariance_matrices.assign(&overall_cov_matrix);

        Ok(Gmm {
            num_gaussians,
            means,
            covariance_matrices,
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

        assert_eq!(gmm.num_gaussians, 2);
        assert_eq!(gmm.means, array![[2.0, 2.0], [2.0, 2.0]]);
        assert_eq!(
            gmm.covariance_matrices,
            array![[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
        );
    }
}
