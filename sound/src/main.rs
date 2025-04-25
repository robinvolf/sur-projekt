use anyhow::{Result, anyhow};
use clap::Parser;
use hound::WavReader;
use ndarray::{self, Array, Array2};
use plotters::prelude::*;
use std::path::{Path, PathBuf};

/// Délka okna jednoho kusu vzorku v ms
const WINDOW_MS: u32 = 50;

#[derive(Parser)]
struct Config {
    #[arg(short)]
    wav_path: PathBuf,
}

/// Vykreslí graf série `series` do souboru `output`, pokud se něco pokazí
/// vrátí Error.
fn plot_series(series: &[i16], output: &Path) -> Result<()> {
    let drawing_area = SVGBackend::new(output, (1920, 1080)).into_drawing_area();
    drawing_area.fill(&WHITE).unwrap();

    let mut chart_builder = ChartBuilder::on(&drawing_area);
    chart_builder
        .margin(20)
        .set_left_and_bottom_label_area_size(20);

    let data_min: i32 = series
        .iter()
        .map(|e| *e)
        .reduce(|acc, e| acc.min(e))
        .unwrap()
        .into();
    let data_max: i32 = series
        .iter()
        .map(|e| *e)
        .reduce(|acc, e| acc.max(e))
        .unwrap()
        .into();

    let mut chart_context = chart_builder
        .build_cartesian_2d(0..series.len(), data_min..data_max)
        .unwrap();

    chart_context.configure_mesh().draw().unwrap();

    chart_context
        .draw_series(
            series
                .iter()
                .enumerate()
                .map(|(index, value)| Circle::new((index, Into::<i32>::into(*value)), 1, &BLUE)),
        )
        .unwrap();

    drawing_area.present().map_err(|err| anyhow!(err))
}

/// Vrátí počet vzorků v okně, pokud vzorkujeme na frekvenci `frequency`.
fn samples_in_window(frequency: f32, window_size_ms: u32) -> usize {
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
fn samples_into_window_matrix(mut samples: Vec<i16>, samples_per_window: usize) -> Array2<i16> {
    // Výpočet dimenzí matice oken
    let samples_in_window = samples_per_window;
    let n_samples = samples.len();
    let n_windows = n_samples / samples_in_window;

    // Ořežeme konec nahrávky tak, abychom mohli vytvořit z vektoru vzorků matici oken
    samples.truncate(n_windows * samples_in_window);

    // Tady pozor, abychom mohli použít paměť vektoru, ndarray předpokládá,
    // že data jsou v row-column order (uložena za sebou po řádcích),
    // tudíž řádek = okno
    Array::from_shape_vec((n_windows, samples_in_window), samples)
        .expect("Počet prvků v matici by měl být roven row * col")
}

fn main() -> Result<()> {
    let config = Config::parse();

    let mut reader = WavReader::open(&config.wav_path).unwrap();
    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();

    let matrix = samples_into_window_matrix(
        samples,
        samples_in_window(reader.spec().sample_rate as f32, WINDOW_MS),
    );

    println!("{:?}", matrix);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_in_window_exact_test() {
        let freq_hz = 200.0;
        assert_eq!(samples_in_window(freq_hz, WINDOW_MS), 10);
    }

    #[test]
    fn samples_in_window_decimal_test() {
        let freq_hz = 204.081632653;
        assert_eq!(
            samples_in_window(freq_hz, WINDOW_MS),
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
