//! Modul pro práci s GMM (Gaussian mixture model).

use std::f32::consts::PI;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, IntoNdProducer, ShapeBuilder, s,
};
use ndarray_linalg::{Determinant, Inverse};
use ndarray_stats::{CorrelationExt, SummaryStatisticsExt};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, num_traits::Inv};

/// Struktura reprezentující generativní model, který modeluje data
/// pomocí směsice Gaussovských rozložení. Data jsou N-dimenzionální vektory.
struct Gmm {
    /// Počet gaussovek, které model používá pro modelování dat
    num_gaussians: usize,
    /// Latentní proměnné určující "váhy" jednotlivých gaussovek. Pokud bych vzorkoval
    /// z tohoto rozdělení, použil bych toto číslo jako pravděpodobnost, že budu vzorkovat
    /// z dané gaussovky.
    latent_variables: Array1<f32>,
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

        let training_data_len = training_data.dim().0;
        let dimensionality = training_data.dim().1;

        let should_terminate = false;
        while !should_terminate {
            // Expectation

            // Pro každé dato a gaussovku spočítám pravděpodobnost, že daná gaussovka vygenerovala dané dato a zváhuju to pravděopdobností, výběru dané gaussovky
            let weighted_probs_from_gaussians = Array2::from_shape_fn(
                (training_data.dim().0, num_gaussians),
                |(data_index, gaussian_index)| {
                    let data = training_data.row(data_index);
                    let mean = gmm.means.row(gaussian_index);
                    let cov = gmm.covariance_matrices.slice(s![gaussian_index, .., ..]);
                    let gauss_prob = gmm.latent_variables[gaussian_index];

                    let prob = Gmm::gauss_multi_variate(mean, cov, data);
                    gauss_prob * prob
                },
            );

            let resp_denom = weighted_probs_from_gaussians.sum_axis(Axis(1));

            // Pro každé dato to říká, jak moc jsou jednotlivé gaussovky za to dato zodpovědné
            let responsibilities = weighted_probs_from_gaussians / &resp_denom.t(); // Musíme transponovat, v responsibilities jsou data po řádcích a gaussovky po sloupcích

            debug_assert_eq!(
                responsibilities.sum_axis(Axis(1)),
                Array1::from_elem(responsibilities.dim().0, 1.0),
                "Suma přes pravděpodobnosti pro každé dato by měla být 1.0"
            );

            // Maximization

            // Váhy jednotlivých gaussovek, když vezmeme v úvahu všechna data (kolik proporčně dat náleží každé z gaussovek)
            let gauss_responsibilities = responsibilities.sum_axis(Axis(0));

            // Spočítáme nové pravděpodobnosti jednotlivých gaussovek
            let new_gauss_probs = gauss_responsibilities / responsibilities.sum();

            // Spočítáme nové střední hodnoty a kovarianční matice
            let weighted_data = &training_data
                .t()
                .broadcast((dimensionality, training_data_len, num_gaussians))
                .expect("Nelze rozšířit trénovací data")
                * &responsibilities
                    .broadcast((dimensionality, training_data_len, num_gaussians))
                    .expect("Nelze rozšířit responsibilities");
            let new_means = weighted_data.sum_axis(Axis(1))
                / new_gauss_probs
                    .broadcast((dimensionality, num_gaussians))
                    .unwrap();

            let new_covs = todo!();
        }

        todo!()
    }

    /// Pravděpodobnost vícerozměrného gaussovského rozložení vstupu `x` při
    /// parametrech `mean` a `cov`.
    fn gauss_multi_variate(mean: ArrayView1<f32>, cov: ArrayView2<f32>, x: ArrayView1<f32>) -> f32 {
        debug_assert!(
            x.dim() == mean.dim() && x.dim() == cov.dim().0 && x.dim() == cov.dim().1,
            "Dimenzionalita dat je jiná než dimenzionalita modelu (data = {}, μ = {}, Σ = {}x{})",
            x.dim(),
            mean.dim(),
            cov.dim().0,
            cov.dim().1
        );

        // Při výpočtu pravděpodobnosti je tato část výpočtu vždy stejná
        let multiplier = (2.0 * PI).powf(-(x.dim() as f32) / 2.0);

        let inv_cov_matrix = cov
            .inv()
            .expect("Nelze spočítat inverzní kovarianční matici");
        let cov_matrix_det = cov
            .det()
            .expect("Nelze spočítat determinant kovarianční matice");
        let exponent = (&x - &mean).dot(&inv_cov_matrix).dot(&(&x - &mean));

        multiplier * cov_matrix_det.inv().sqrt() * exponent.exp()
    }

    /// Spočítá pravděpodobnost data `x` při aktuálním nastavení modelu.
    /// Pokud nesedí dimenzionalita dat a modelu, vrátí Error.
    fn get_prob(&self, x: ArrayView1<f32>) -> Result<f32> {
        let data_dimensionality = x.dim();
        let model_dimensionality = self.means.dim().1;

        if data_dimensionality != model_dimensionality {
            Err(anyhow!(
                "Dimenzionalita dat je jiná než dimenzionalita modelu ({data_dimensionality} ≠ {model_dimensionality})"
            ))
        } else {
            // Pravděpodobnosti tohoto data z jednotlivých gaussovek
            let partial_probs = Array1::from_iter(
                self.means
                    .axis_iter(Axis(0))
                    .zip(self.covariance_matrices.axis_iter(Axis(0)))
                    .map(|(mean, cov)| Gmm::gauss_multi_variate(mean, cov, x)),
            );

            debug_assert!(
                partial_probs
                    .iter()
                    .fold(true, |acc, x| acc && *x >= 0.0 && *x <= 1.0),
                "Dílčí pravděpodobnosti musí být hodnoty mezi 0 a 1!"
            );

            // Skalární součin dílčích pravděpodobností a pravděpodobností výběru
            let prob = partial_probs.dot(&self.latent_variables);

            Ok(prob)
        }
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

        let latent_variables = Array1::<f32>::from_elem(num_gaussians, 1.0 / num_gaussians as f32);

        Ok(Gmm {
            num_gaussians,
            latent_variables,
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
