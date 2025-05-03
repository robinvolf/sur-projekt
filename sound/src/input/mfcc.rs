//! Modul pro zpracování audia pomocí MFCC.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2, s};
use ndrustfft::{DctHandler, R2cFftHandler, nddct2, ndfft_r2c};
use rand_distr::{Distribution, Normal};
use std::{cmp::min, f64::consts::PI};

const MEL_FILTER_START_FREQ: usize = 20;
const MEL_FILTER_END_FREQ: usize = 8000;

pub struct MFCCSettings {
    /// Velikost okna v ms
    pub window_size_ms: u32, // = 100;

    /// Počet vzorků, o které se budou okna překrývat
    pub windows_overlap: usize, // = 400;

    /// Počet Mel-filter bank, které budou aplikovány na signál
    pub num_mel_filter_banks: usize, // = 100;

    /// Maximální počet koeficientů
    pub atmost_coeffs: usize,
}

/// Vrátí počet vzorků v okně, pokud vzorkujeme na frekvenci `frequency`.
pub fn samples_in_window(frequency: f64, window_size_ms: u32) -> usize {
    let window_secs: f64 = window_size_ms as f64 / 1000.0;

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
    samples: &[f64],
    samples_per_window: usize,
    overlap: usize,
) -> Array2<f64> {
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

/// Získá Mel-frequancy kepstral koeficienty pro zadané vzorky `samples`, kde:
///   - `sampling_freq` je vzorkovací frekvence se kterou byly vzorky pořízeny
///   - `window_overlap` je počet vzorků, o které se jednotlivá okna signálu budou překrývat
///   - `samples_in_window` je počet vzorků, který bude každé okno obsahovat
///   - `mel_filter_banks` je počet bank filtrů, které se na signál aplikují (čím víc, tím větší dimenzionalita výstupních MFCC koeficientů jednotlivých oken).
///
/// ### Jak to funguje?
/// Výpočet MFCC koeficientů obnáší tyto kroky:
/// 1. Zanesení slabého gaussovského šumu do signálu, abychom se vyhli pozdějším možným numerickým komplikacím
/// 2. Rozdělíme signál na překrývající se okna
///   - Když počet oken nesedí přesně na počet vzorků v signálu, signál se ořeže
/// 3. Aplikujeme Tukeyho okno na okna signálu, abychom je na začátku a na konci utlumili do nuly
/// 4. Spočteme FFT oken signálu a ponecháme si pouze amplitudovou složku
/// 5. Aplikujeme banku Mel-filtrů na amplitudy oken
/// 6. Zlogaritmujeme
/// 7. Spočteme kosinovou transformaci
/// 8. Vybereme nejvýše `atmost_coeffs` koeficientů pro každé okno
pub fn mfcc(samples: &[f64], sampling_frequency: usize, settings: &MFCCSettings) -> Array2<f64>
where
{
    let samples_in_window = samples_in_window(sampling_frequency as f64, settings.window_size_ms);

    // Vneseme do vzorku Gaussovský šum kolem nuly, abychom se vyhli numerickým problémům při logaritmování
    let noise = Normal::new(0.0, 1.0)
        .unwrap()
        .sample_iter(rand::rng())
        .take(samples.len());

    let samples: Vec<f64> = samples
        .iter()
        .zip(noise)
        .map(|(sample, noise)| *sample + noise)
        .collect();

    // Rozdělíme vzorky na okna
    let mut windows_matrix =
        samples_into_window_matrix(&samples, samples_in_window, settings.windows_overlap);

    // Aplikujeme Tukeyho okno na všechny vzorky
    apply_tukey_window(windows_matrix.view_mut());

    // Aplikujeme FFT a ponecháme si pouze amplitudy
    let windows_fft = fft_only_amplitudes(&windows_matrix);

    // Aplikujeme mel-banku filtrů
    let mel_coefs = apply_mel_filter_bank(
        &windows_fft,
        MEL_FILTER_START_FREQ,
        MEL_FILTER_END_FREQ,
        settings.num_mel_filter_banks,
        sampling_frequency,
    );

    // Zlogaritmování
    let log_mel_coefs = mel_coefs.ln();

    // Diskrétní kosinová transformace pro jednotlivé řádky
    let dct_handler = DctHandler::<f64>::new(log_mel_coefs.shape()[1]);
    let mut mfcc_coeffs = Array2::zeros(log_mel_coefs.dim());
    nddct2(&log_mel_coefs, &mut mfcc_coeffs, &dct_handler, 1);

    let num_of_mfcc_coeffs_per_window = mfcc_coeffs.dim().1;

    mfcc_coeffs.slice_move(s![
        ..,
        0..min(num_of_mfcc_coeffs_per_window, settings.atmost_coeffs)
    ])
}

/// Zkonvertuje frekvenci `x` do Mel-scale
fn mel_scale(x: f64) -> f64 {
    1125.0 * (1.0 + x / 700.0).ln()
}

/// Zkonvertuje číslo `x` v Mel-scale na frekvenci
fn mel_scale_inv(x: f64) -> f64 {
    700.0 * ((x / 1125.0).exp() - 1.0)
}

/// Aplikuje Mel-filtr banky na jednotlivá okna.
/// Vytvoření Mel-filtr bank převzato z: <http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs>
fn apply_mel_filter_bank(
    windows: &Array2<f64>,
    freq_start: usize,
    freq_end: usize,
    num_banks: usize,
    samples_rate: usize,
) -> Array2<f64> {
    let mel_start = mel_scale(freq_start as f64);
    let mel_end = mel_scale(freq_end as f64);
    let window_len = windows.dim().1;

    let mut filter_bank_points = Array1::linspace(mel_start, mel_end, num_banks + 2); // +2 protože ještě musíme započítat začátek a konec

    // Převedem zpátky do domény frekvencí, nyní jsou frekvence nelineárně rozmístěny
    filter_bank_points.map_inplace(|x| *x = mel_scale_inv(*x));

    // Převod na FFT bins
    filter_bank_points
        .map_inplace(|x| *x = ((2 * window_len + 1) as f64 * *x / samples_rate as f64).floor());

    let banks = Array2::from_shape_fn((num_banks, window_len), |(m, k)| {
        let m = m + 1; // M-kem se jen indexuju do filter_bank_points, které jsou o jeden prvek na každou stranu větší, než výsledná banka filtrů
        let k = k as f64;

        if k < filter_bank_points[m - 1] || k > filter_bank_points[m + 1] {
            0.0
        } else if k >= filter_bank_points[m - 1] && k <= filter_bank_points[m] {
            (k - filter_bank_points[m - 1]) / (filter_bank_points[m] - filter_bank_points[m - 1])
        } else {
            (filter_bank_points[m + 1] - k) / (filter_bank_points[m + 1] - filter_bank_points[m])
        }
    });

    // Aplikuju filtry na jednotlivá a sečtu aplikace filtru na okno
    let applied_mel_filters = Array2::from_shape_fn(
        (windows.shape()[0], num_banks),
        |(window_index, bank_index)| {
            let filtered_window = &windows.row(window_index) * &banks.row(bank_index);
            filtered_window.sum()
        },
    );

    applied_mel_filters
}

/// Aplikuje FFT na jednotlivá okna (řádky matice).
/// Výstupní matice bude mít stejný počet oken, ale budou menší
/// (konkrétně n/2 + 1), kvůli symetrii FFT při zpracování reálného signálu.
fn fft_only_amplitudes(windows: &Array2<f64>) -> Array2<f64> {
    let window_len = windows.dim().1;
    let fft_handler = R2cFftHandler::<f64>::new(window_len);

    let mut fft_complex = Array2::zeros((windows.dim().0, window_len / 2 + 1));

    // FFT přes jednotlivá okna, výstup jsou komplexní čísla
    // Jelikož je signál diskrétní, nová okna budou mít poloviční velikost (kvůli symetrii)
    ndfft_r2c(windows, &mut fft_complex, &fft_handler, 1);

    // Matice, kde řádky jsou reálná čísla reprezentující amplitudy z FFT
    let fft_only_amplitude = Array2::from_shape_fn(fft_complex.dim(), |(i, j)| {
        let complex = fft_complex[[i, j]];
        (complex.re.powi(2) + complex.im.powi(2)).sqrt()
    });

    fft_only_amplitude
}

/// Aplikuje Tukeyho okno na matici vzorků. Toto okno utlumuje na začátku a na konci,
/// ale prostřední část signálu, nechává nedotčenou.
fn apply_tukey_window(mut windows: ArrayViewMut2<f64>) {
    // Převzato z https://en.wikipedia.org/wiki/Window_function#Examples_of_window_functions
    const ALPHA: f64 = 0.2;
    const TWO_PI: f64 = 2.0 * PI;

    let window_len = windows.dim().1;

    let tukey_window = Array1::from_iter((0..windows.dim().1).into_iter().map(|i| {
        if (i as f64) < ALPHA * window_len as f64 / 2.0
            || (i as f64) > window_len as f64 * (2.0 - ALPHA) / 2.0
        {
            0.5 * (1.0 - f64::cos((TWO_PI * i as f64) / (ALPHA * window_len as f64)))
        } else {
            1.0
        }
    }));

    // tukey_window udělá broadcast a prvek po prvku se vynásobí se vzorky v každém okně
    windows *= &tukey_window;
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

        let expected_windows = array![[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];

        const TOLERANCE: f64 = 0.01;

        // assert_eq!(windows, expected_windows) nefunguje, kvůli nepřesnosti floatů
        for (got, expected) in windows.into_iter().zip(expected_windows.into_iter()) {
            assert!((got - expected).abs() < TOLERANCE);
        }
    }
}
