use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use example_utils::*;
use int_data_analysis::kmeans::{KMeans, Model};
use ndarray::{s, Array2, Axis};
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/wine-quality.csv")?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    println!("Clusters - Inertia");
    let mut models: HashMap<u32, Model> = HashMap::new();
    for n_clusters in 1..=6 {
        let model = KMeans::default().n_clusters(n_clusters).fit(&data);
        println!("{n_clusters} - {}", model.inertia());
        models.insert(n_clusters, model);
    }

    let best_model = models.get(&4).unwrap();
    println!("Result\n{:?}", best_model.centroids());

    let now = Utc::now().format("(%H:%M:%S %d.%m.%Y)").to_string();
    let filepath = format!("figures/wine/clustering {}.svg", now);
    create_dir_all("figures/wine")?;
    create_plot(filepath, best_model, &data)?;
    Ok(())
}

fn create_plot(filepath: String, model: &Model, data: &Array2<f64>) -> Result<(), Box<dyn Error>> {
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

    let root = SVGBackend::new(&filepath, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart_ctx = ChartBuilder::on(&root)
        .margin(10)
        .margin_right(30)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..model.centroids().ncols() - 1, 0f64..1f64)?;

    chart_ctx
        .configure_mesh()
        .x_labels(model.centroids().ncols())
        .y_labels(8)
        .light_line_style(WHITE.mix(0.3))
        .x_label_formatter(&|n| String::from(x_labels[*n]))
        .draw()?;

    let norm_data = normalize_centroids(&model.centroids(), data);
    for (idx, each) in norm_data.rows().into_iter().enumerate() {
        chart_ctx.draw_series(LineSeries::new(
            each.iter().enumerate().map(|(x, y)| (x, *y)),
            Palette99::pick(idx),
        ))?;
        chart_ctx.draw_series(
            each.iter()
                .enumerate()
                .map(|(x, y)| Circle::new((x, *y), 3, Palette99::pick(idx).filled())),
        )?;
    }

    root.present()?;
    Ok(())
}

fn normalize_centroids(centroids: &Array2<f64>, data: &Array2<f64>) -> Array2<f64> {
    let mut temp = data.clone();
    temp.append(Axis(0), centroids.view())
        .expect("shapes should match");
    let normalized = normalize_data(&temp);
    normalized
        .slice(s![normalized.nrows() - centroids.nrows().., ..])
        .into_owned()
}
