use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use example_utils::*;
use int_data_analysis::*;
use ndarray::{s, Array2, Axis};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashMap;
use std::error::Error;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn Error>> {
    let data = parse_file("data/wine-quality.csv")?
        .slice(s![.., ..2usize])
        .to_owned();
    let (train, test) = split_train_test(data, 0.25);
    let class_markers = clusterize_and_predict(&train);
    let knn = KNearest::new(11, train.clone(), class_markers);
    let now = Utc::now().format("(%H:%M:%S %d.%m.%Y)").to_string();
    create_dir_all("figures/wine")?;
    create_plot(format!("figures/wine/knn {}.svg", now), &train, &test, &knn)?;
    Ok(())
}

fn parse_file(path: &str) -> Result<Array2<f64>, csv::Error> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    records.shuffle(&mut Xoshiro256Plus::seed_from_u64(42));
    Ok(records_into_array(&records))
}

fn clusterize_and_predict(data: &Array2<f64>) -> Vec<usize> {
    let mut models: HashMap<u32, Model> = HashMap::new();
    for n_clusters in 1..=6 {
        let model = KMeans::default().n_clusters(n_clusters).fit(data);
        models.insert(n_clusters, model);
    }
    let best_model = models.get(&4).unwrap();

    let row_iter = data.axis_iter(Axis(0));
    row_iter.map(|p| best_model.predict(p) + 1).collect()
}

pub fn create_plot(
    filepath: String,
    train: &Array2<f64>,
    test: &Array2<f64>,
    knn: &KNearest,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(2, test.ncols(), "{}", "wrong shape to build scatter plot");

    let root = SVGBackend::new(&filepath, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut temp = train.clone();
    temp.append(Axis(0), test.view())?;

    let ranges = calculate_ranges_2d(&temp).expect("Should not be empty");
    let mut scatter_ctx = ChartBuilder::on(&root)
        .caption(
            "Wine - Alcohol x Malic Acid",
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(ranges.0, ranges.1)?;

    scatter_ctx.configure_mesh().disable_mesh().draw()?;
    plot_data(&mut scatter_ctx, test, knn, |s| s.filled())?;
    plot_data(&mut scatter_ctx, train, knn, |s| s.stroke_width(1))?;
    root.present()?;

    Ok(())
}

fn plot_data(
    scatter_ctx: &mut ChartContext<SVGBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    data: &Array2<f64>,
    knn: &KNearest,
    style_fn: fn(s: ShapeStyle) -> ShapeStyle,
) -> Result<(), Box<dyn Error>> {
    scatter_ctx.draw_series(data.axis_iter(Axis(0)).map(|row| {
        let style = map_class_to_color(knn.predict(row));
        Circle::new((row[0], row[1]), 3, style_fn(style))
    }))?;
    Ok(())
}
