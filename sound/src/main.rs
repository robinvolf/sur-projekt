use anyhow::{Result, anyhow};
use clap::Parser;
use plotters::prelude::*;
use std::path::{Path, PathBuf};

mod input;

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

fn main() -> Result<()> {
    let config = Config::parse();

    let matrix = input::load_wav_file_16bit(&config.wav_path, 50)?;

    println!("{:?}", matrix);

    Ok(())
}
