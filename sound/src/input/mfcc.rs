//! Modul pro zpracování audia pomocí MFCC.

use ndarray::{Array2, ArrayView1};

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
    samples: &[i16],
    samples_per_window: usize,
    overlap: usize,
) -> Array2<i16> {
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

pub fn mfcc(samples: &[i16], samples_per_window: usize, window_overlap: usize) -> Array2<i16> {
    // TODO: Vnést šum do vzorků?

    // Rozdělíme vzorky na okna
    let windows_matrix = samples_into_window_matrix(samples, samples_per_window, window_overlap);

    todo!()
}

#[cfg(test)]
mod tests {
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
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];

        let matrix = samples_into_window_matrix(&samples, 4, 0);

        assert_eq!(matrix[[0, 0]], 1);
        assert_eq!(matrix[[4, 3]], 20);
        assert_eq!(matrix[[0, 3]], 4);
        assert_eq!(matrix[[4, 0]], 17);
    }

    #[test]
    fn samples_into_window_matrix_truncation_test() {
        let samples = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ];

        // Počet samples není dělitelný 4, ale jejich 23 -> tři vzorky by měly být zahozeny
        let matrix = samples_into_window_matrix(&samples, 4, 0);

        assert_eq!(matrix[[0, 0]], 1);
        assert_eq!(matrix[[4, 3]], 20);
        assert_eq!(matrix[[0, 3]], 4);
        assert_eq!(matrix[[4, 0]], 17);
    }

    #[test]
    fn samples_into_window_matrix_overlap_test() {
        let samples = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Overlap = 1, tudíž jednotlivé řádky budou
        // 1, 2, 3, 4
        // 3, 4, 5, 6,
        // 5, 6, 7, 8,
        // 7, 8, 9, 10,
        let matrix = samples_into_window_matrix(&samples, 4, 2);

        assert_eq!(matrix[[0, 0]], 1);
        assert_eq!(matrix[[3, 3]], 10);

        assert_eq!(matrix[[0, 2]], matrix[[1, 0]]);
        assert_eq!(matrix[[0, 3]], matrix[[1, 1]]);

        assert_eq!(matrix[[3, 3]], 10)
    }

    #[test]
    fn samples_into_window_matrix_overlap_truncation_test() {
        let samples = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
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

        assert_eq!(matrix[[0, 0]], 1);
        assert_eq!(matrix[[3, 3]], 10);

        assert_eq!(matrix[[0, 2]], matrix[[1, 0]]);
        assert_eq!(matrix[[0, 3]], matrix[[1, 1]]);

        assert_eq!(matrix[[3, 3]], 10)
    }
}
