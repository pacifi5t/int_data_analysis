use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use int_data_analysis::kmeans::{KMeans, Model};
use int_data_analysis::utils::records_into_array;
use linfa::prelude::Fit;
use linfa::DatasetBase;
use ndarray::Array2;
use plotters::backend::BitMapBackend;
use plotters::prelude::{ChartBuilder, Color, IntoDrawingArea, IntoFont, WHITE};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/wine-quality.csv")?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    let n_clusters = 3;
    let model = KMeans::default().n_clusters(n_clusters).fit(&data);
    println!("Result\n{:?}", model.centroids());

    let now = Utc::now().format("(%Y-%m-%d %H:%M:%S)").to_string();
    let filepath = format!("figures/examples/wine_quality {}.png", now);
    create_plot("Wine quality", filepath, &data, &model)?;

    let linfa_model =
        linfa_clustering::KMeans::params(n_clusters as usize).fit(&DatasetBase::from(data))?;
    println!("Linfa\n{:?}", linfa_model.centroids());

    Ok(())
}

fn create_plot(
    caption: &str,
    filepath: String,
    data: &Array2<f64>,
    model: &Model,
) -> Result<(), Box<dyn Error>> {
    let x_labels = [
        "Alcohol",
        "Malic Acid",
        "Ash",
        "Ash Alcanity",
        "Magnesium",
        "Total Phenols",
        "Flavanoids",
        "Nonflavanoid Phenols",
        "Proanthocyanins",
        "Color Intensity",
        "Hue",
        "OD280",
        "Proline",
    ];

    let root = BitMapBackend::new(&filepath, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart_ctx = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(10)
        .margin_right(30)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..model.centroids().ncols() - 1, 0f64..1f64)?;

    chart_ctx
        .configure_mesh()
        .x_labels(model.centroids().ncols())
        .y_labels(8)
        .light_line_style(&WHITE.mix(0.3))
        .x_label_formatter(&|n| String::from(x_labels[*n]))
        .draw()?;

    //TODO: Plot data

    root.present()?;
    Ok(())
}
