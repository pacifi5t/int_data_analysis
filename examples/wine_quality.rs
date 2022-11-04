use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use int_data_analysis::kmeans::{KMeans, Model};
use int_data_analysis::utils::records_into_array;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use plotters::backend::BitMapBackend;
use plotters::prelude::{
    ChartBuilder, Circle, Color, IntoDrawingArea, IntoFont, LineSeries, ShapeStyle, BLACK, BLUE,
    CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW,
};
use std::collections::HashMap;
use std::error::Error;

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

    let now = Utc::now().format("(%Y-%m-%d %H:%M:%S)").to_string();
    let filepath = format!("figures/examples/wine_quality {}.png", now);
    create_plot("Wine quality", filepath, best_model, &data)?;
    Ok(())
}

fn create_plot(
    caption: &str,
    filepath: String,
    model: &Model,
    data: &Array2<f64>,
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

    let norm_data = normalize_centroids(&model.centroids(), &data);
    let mut ctr = 0;
    for each in norm_data.rows() {
        chart_ctx.draw_series(LineSeries::new(
            each.iter().enumerate().map(|(x, y)| (x, *y)),
            map_index_to_color(ctr),
        ))?;
        chart_ctx.draw_series(
            each.iter()
                .enumerate()
                .map(|(x, y)| Circle::new((x, *y), 3, map_index_to_color(ctr).filled())),
        )?;
        ctr += 1;
    }

    root.present()?;
    Ok(())
}

fn normalize_centroids(centroids: &Array2<f64>, data: &Array2<f64>) -> Array2<f64> {
    let mut temp = data.clone();
    temp.append(Axis(0), centroids.view())
        .expect("Should be fine");
    let normalized = normalize_data(&temp);
    normalized
        .slice(s![normalized.nrows() - centroids.nrows().., ..])
        .into_owned()
}

fn normalize_data(data: &Array2<f64>) -> Array2<f64> {
    let mut array: Array2<f64> = Array2::zeros((data.nrows(), 0));
    for c in data.columns() {
        let min = min(c);
        let max = max(c);
        let v: Vec<f64> = c
            .into_iter()
            .map(|each| (each - min) / (max - min))
            .collect();
        array
            .push_column(Array1::from(v).view())
            .expect("Column length match");
    }
    array
}

fn min(c: ArrayView1<f64>) -> f64 {
    let mut min = f64::MAX;
    for each in c {
        if *each < min {
            min = *each;
        }
    }
    min
}

fn max(c: ArrayView1<f64>) -> f64 {
    let mut max = f64::MIN;
    for each in c {
        if *each > max {
            max = *each;
        }
    }
    max
}

fn map_index_to_color(index: usize) -> ShapeStyle {
    let color = match index {
        0 => RED,
        1 => GREEN,
        2 => BLUE,
        3 => MAGENTA,
        4 => CYAN,
        5 => YELLOW,
        _ => BLACK,
    };
    color.into()
}
