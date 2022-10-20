use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use int_data_analysis::kmeans::KMeans;
use int_data_analysis::utils::records_into_array;
use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let n_clusters = 3;

    let filepath = "data/clusters.csv";
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(filepath)?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    let model = KMeans::default().n_clusters(n_clusters).fit(&data);
    println!("Result {:?}", model.centroids());

    let now = Utc::now().format("(%Y-%m-%d %H:%M:%S)").to_string();
    let filename = format!("figures/examples/my_kmeans {}.png", now);
    let root = BitMapBackend::new(&filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("My KMeans", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-30f32..30f32, -30f32..30f32)?;

    chart.configure_mesh().draw()?;
    root.present()?;

    Ok(())
}
