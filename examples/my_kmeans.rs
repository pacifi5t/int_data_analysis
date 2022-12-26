use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use example_utils::*;
use int_data_analysis::*;
use ndarray::{Array2, Axis};
use plotters::prelude::*;
use std::error::Error;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/clusters.csv")?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    let n_clusters = 3;
    let model = KMeans::default().n_clusters(n_clusters).fit(&data);
    println!("Result\n{:?}", model.centroids());

    let now = Utc::now().format("(%H:%M:%S %d.%m.%Y)").to_string();
    let filepath = format!("figures/test/my-kmeans {}.svg", now);
    create_dir_all("figures/test")?;
    plot_clusters(filepath, &data, &model)?;

    Ok(())
}

pub fn plot_clusters(
    filepath: String,
    data: &Array2<f64>,
    model: &Model,
) -> Result<(), Box<dyn Error>> {
    let error_msg = "wrong shape to build scatter plot";
    assert_eq!(2, data.ncols(), "{}", error_msg);
    assert_eq!(2, model.centroids().ncols(), "{}", error_msg);

    let root = SVGBackend::new(&filepath, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let ranges = calculate_ranges_2d(data).expect("should not be empty");
    let mut scatter_ctx = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(ranges.0, ranges.1)?;

    scatter_ctx.configure_mesh().disable_mesh().draw()?;
    scatter_ctx.draw_series(data.axis_iter(Axis(0)).map(|row| {
        let style = Palette99::pick(model.predict(row));
        Circle::new((row[0], row[1]), 2, style.filled())
    }))?;
    root.present()?;

    Ok(())
}
