//! Modul pro zpracování audia pomocí MFCC.

use ndarray::Array2;

/// Vrátí počet vzorků v okně, pokud vzorkujeme na frekvenci `frequency`.
pub fn samples_in_window(frequency: f32, window_size_ms: u32) -> usize {
    let window_secs: f32 = window_size_ms as f32 / 1000.0;

    let period = 1.0 / frequency;

    (window_secs / period) as usize
}

/// Vytvoří ze `samples` matici, kde každý řádek reprezentuje jedno okno vzorků
/// o délce `samples_per_window`. Pokud je vzorků víc než `počet oken * vzorků na okno`
/// budou přebývající vzorky zahozeny (nejvíce se zahodí `vzorků na okno - 1`).
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
pub fn samples_into_window_matrix(mut samples: Vec<i16>, samples_per_window: usize) -> Array2<i16> {
    // Výpočet dimenzí matice oken
    let samples_in_window = samples_per_window;
    let n_samples = samples.len();
    let n_windows = n_samples / samples_in_window;

    // Ořežeme konec nahrávky tak, abychom mohli vytvořit z vektoru vzorků matici oken
    samples.truncate(n_windows * samples_in_window);

    // Tady pozor, abychom mohli použít paměť vektoru, ndarray předpokládá,
    // že data jsou v row-column order (uložena za sebou po řádcích),
    // tudíž řádek = okno
    Array2::from_shape_vec((n_windows, samples_in_window), samples)
        .expect("Počet prvků v matici by měl být roven row * col")
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

        let matrix = samples_into_window_matrix(samples, 4);

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
        let matrix = samples_into_window_matrix(samples, 4);

        assert_eq!(matrix[[0, 0]], 1);
        assert_eq!(matrix[[4, 3]], 20);
        assert_eq!(matrix[[0, 3]], 4);
        assert_eq!(matrix[[4, 0]], 17);
    }
}
