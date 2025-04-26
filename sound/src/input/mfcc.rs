//! Modul pro zpracování audia pomocí MFCC.

use std::f32::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2};
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};

/// Vrátí počet vzorků v okně, pokud vzorkujeme na frekvenci `frequency`.
pub fn samples_in_window(frequency: f32, window_size_ms: u32) -> usize {
    let window_secs: f32 = window_size_ms as f32 / 1000.0;

    let period = 1.0 / frequency;

    (window_secs / period) as usize
}

/// Vytvoří ze `samples` matici, kde každý řádek reprezentuje jedno okno vzorků
/// o délce `samples_per_window`. Okna se překrývají o `overlap` vzorků.
/// Pokud je vzorků víc než je třeba, neprobíhá žádný padding, ale vzorky se jednoduše zahodí.
///
/// ### Tvar matice
/// Matice je tohoto tvaru, kde:
///   - `K` = Počet vzorků v jednom okně
///   - `N` = Počet jednotlivých oken
/// ```
///    K
/// [ ... ]
/// [ ... ]
/// [ ... ] N
/// [ ... ]
/// [ ... ]
/// ```
pub fn samples_into_window_matrix(
    samples: &[f32],
    samples_per_window: usize,
    overlap: usize,
) -> Array2<f32> {
    // Výpočet dimenzí matice oken
    let samples_in_window = samples_per_window;

    let mut matrix = Array2::zeros((0, samples_in_window)); // 0 rows = prázdná matice

    // Na překrývající se okna iterátory nepostačí, musíme manuálně
    let mut index = 0;
    while index + samples_in_window <= samples.len() {
        let window_samples = &samples[index..(index + samples_in_window)];
        matrix
            .push_row(ArrayView1::from(window_samples))
            .expect("Vzorky by měly mít stejnou délku jako řádky matice");

        index += samples_in_window - overlap;
        if index + samples_in_window > samples.len() {
            break;
        }
    }

    matrix
}

/// Získá Mel-frequancy kepstral koeficienty pro zadané vzorky `samples`.
/// TODO: Jak to funguje
pub fn mfcc(samples: &[f32], samples_per_window: usize, window_overlap: usize) -> Array2<f32>
where
{
    // Vneseme do vzorku Gaussovský šum kolem nuly, abychom se vyhli numerickým problémům při logaritmování
    let noise = Normal::new(0.0, 1.0)
        .unwrap()
        .sample_iter(rand::rng())
        .take(samples.len());

    let samples: Vec<f32> = samples
        .iter()
        .zip(noise)
        .map(|(sample, noise)| *sample + noise)
        .collect();

    // Rozdělíme vzorky na okna
    let mut windows_matrix =
        samples_into_window_matrix(&samples, samples_per_window, window_overlap);

    // Aplikujeme Tukeyho okno na všechny vzorky
    apply_tukey_window(windows_matrix.view_mut());

    todo!()
}

/// Aplikuje Tukeyho okno na matici vzorků. Toto okno utlumuje na začátku a na konci,
/// ale prostřední část signálu, nechává nedotčenou.
fn apply_tukey_window(mut windows: ArrayViewMut2<f32>) {
    // Převzato z https://en.wikipedia.org/wiki/Window_function#Examples_of_window_functions
    const ALPHA: f32 = 0.2;
    const TWO_PI: f32 = 2.0 * PI;

    let window_len = windows.dim().1;

    let tukey_window = Array1::from_iter((0..windows.dim().1).into_iter().map(|i| {
        if (i as f32) < ALPHA * window_len as f32 / 2.0
            || (i as f32) > window_len as f32 * (2.0 - ALPHA) / 2.0
        {
            0.5 * (1.0 - f32::cos((TWO_PI * i as f32) / (ALPHA * window_len as f32)))
        } else {
            1.0
        }
    }));

    // tukey_window udělá broadcast a prvek po prvku se vynásobí se vzorky v každém okně
    windows *= &tukey_window;
}

/// Vykreslí graf série `series` do souboru `output`, pokud se něco pokazí
/// vrátí Error.
use anyhow::{Result, anyhow};
use plotters::prelude::*;
use std::path::Path;
fn plot_series(series: &[f32], output: &Path) -> Result<()> {
    let drawing_area = SVGBackend::new(output, (1920, 1080)).into_drawing_area();
    drawing_area.fill(&WHITE).unwrap();

    let mut chart_builder = ChartBuilder::on(&drawing_area);
    chart_builder
        .margin(20)
        .set_left_and_bottom_label_area_size(20);

    let data_min = series
        .iter()
        .map(|e| *e)
        .reduce(|acc, e| acc.min(e))
        .unwrap();
    let data_max = series
        .iter()
        .map(|e| *e)
        .reduce(|acc, e| acc.max(e))
        .unwrap();

    let mut chart_context = chart_builder
        .build_cartesian_2d(0..series.len(), data_min..data_max)
        .unwrap();

    chart_context.configure_mesh().draw().unwrap();

    chart_context
        .draw_series(LineSeries::new(
            series
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value)),
            &BLUE,
        ))
        .unwrap();

    drawing_area.present().map_err(|err| anyhow!(err))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    const TEST_WINDOWS_MS: u32 = 50;

    #[test]
    fn samples_in_window_exact_test() {
        let freq_hz = 200.0;
        assert_eq!(samples_in_window(freq_hz, TEST_WINDOWS_MS), 10);
    }

    #[test]
    fn samples_in_window_decimal_test() {
        let freq_hz = 204.081632653;
        assert_eq!(
            samples_in_window(freq_hz, TEST_WINDOWS_MS),
            10,
            "Frekvence je trochu větší, tudíž samples vyjde 10.něco, ale já pořád chci míň, jen 10"
        );
    }

    #[test]
    fn samples_into_window_matrix_test() {
        let samples = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];

        let matrix = samples_into_window_matrix(&samples, 4, 0);

        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[4, 3]], 20.0);
        assert_eq!(matrix[[0, 3]], 4.0);
        assert_eq!(matrix[[4, 0]], 17.0);
    }

    #[test]
    fn samples_into_window_matrix_truncation_test() {
        let samples = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
        ];

        // Počet samples není dělitelný 4, ale jejich 23 -> tři vzorky by měly být zahozeny
        let matrix = samples_into_window_matrix(&samples, 4, 0);

        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[4, 3]], 20.0);
        assert_eq!(matrix[[0, 3]], 4.0);
        assert_eq!(matrix[[4, 0]], 17.0);
    }

    #[test]
    fn samples_into_window_matrix_overlap_test() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // Overlap = 1, tudíž jednotlivé řádky budou
        // 1, 2, 3, 4
        // 3, 4, 5, 6,
        // 5, 6, 7, 8,
        // 7, 8, 9, 10,
        let matrix = samples_into_window_matrix(&samples, 4, 2);

        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[3, 3]], 10.0);

        assert_eq!(matrix[[0, 2]], matrix[[1, 0]]);
        assert_eq!(matrix[[0, 3]], matrix[[1, 1]]);

        assert_eq!(matrix[[3, 3]], 10.0)
    }

    #[test]
    fn samples_into_window_matrix_overlap_truncation_test() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        // Overlap = 1, tudíž jednotlivé řádky budou
        // 1, 2, 3, 4
        // 3, 4, 5, 6,
        // 5, 6, 7, 8,
        // 7, 8, 9, 10,
        //
        // další řádek by měl být
        // 9, 10, 11, X
        // ale protože končí data u 11ky, měla by se zahodit
        let matrix = samples_into_window_matrix(&samples, 4, 2);

        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[3, 3]], 10.0);

        assert_eq!(matrix[[0, 2]], matrix[[1, 0]]);
        assert_eq!(matrix[[0, 3]], matrix[[1, 1]]);

        assert_eq!(matrix[[3, 3]], 10.0)
    }

    #[test]
    fn apply_tukey_window_test() {
        // Matice s jediným řádkem se samými jedničkami
        let mut windows = array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
        apply_tukey_window(windows.view_mut());

        let expected_windows = array![[0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.49999967]];

        const TOLERANCE: f32 = 0.01;

        // assert_eq!(windows, expected_windows) nefunguje, kvůli nepřesnosti floatů
        for (got, expected) in windows.into_iter().zip(expected_windows.into_iter()) {
            assert!((got - expected).abs() < TOLERANCE);
        }
    }
}
