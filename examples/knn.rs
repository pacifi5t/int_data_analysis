use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use example_utils::*;
use int_data_analysis::*;
use ndarray::{Array2, Axis};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::error::Error;
use std::fs::create_dir_all;

fn main() -> Result<(), Box<dyn Error>> {
    let clusters = 3;
    let train = parse_file("data/clusters.csv")?;
    let test = parse_file("data/clusters-test.csv")?;
    let class_markers = clusterize_and_predict(clusters, &train);

    let knn = KNearest::new(3, train.clone(), class_markers);

    let now = Utc::now().format("(%H:%M:%S %d.%m.%Y)").to_string();
    create_dir_all("figures/test")?;
    create_plot(format!("figures/test/knn {}.svg", now), &train, &test, &knn)?;
    Ok(())
}

fn parse_file(path: &str) -> Result<Array2<f64>, csv::Error> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    Ok(records_into_array(&records))
}

fn clusterize_and_predict(clusters: u32, data: &Array2<f64>) -> Vec<usize> {
    let model = KMeans::default().n_clusters(clusters).fit(data);
    let row_iter = data.axis_iter(Axis(0));
    row_iter.map(|p| model.predict(p) + 1).collect()
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
        .margin(5)
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
        let style = style_fn(Palette99::pick(knn.predict(row) - 1).into());
        Circle::new((row[0], row[1]), 3, style)
    }))?;
    Ok(())
}
