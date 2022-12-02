use chrono::Utc;
use csv::{ReaderBuilder, StringRecord};
use example_utils::records_into_array;
use int_data_analysis::kmeans::{KMeans, Model};
use ndarray::{Array2, Axis};
use plotters::prelude::*;
use std::error::Error;
use std::ops::Range;

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/clusters.csv")?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    let n_clusters = 3;
    let model = KMeans::default().n_clusters(n_clusters).fit(&data);
    println!("Result\n{:?}", model.centroids());

    let now = Utc::now().format("(%Y-%m-%d %H:%M:%S)").to_string();
    let filepath = format!("figures/examples/my_kmeans {}.png", now);
    plot_clusters("My Kmeans", filepath, &data, &model)?;

    Ok(())
}

pub fn plot_clusters(
    caption: &str,
    filepath: String,
    data: &Array2<f64>,
    model: &Model,
) -> Result<(), Box<dyn Error>> {
    let error_msg = "Wrong shape to build scatter plot";
    assert_eq!(2, data.ncols(), "{}", error_msg);
    assert_eq!(2, model.centroids().ncols(), "{}", error_msg);

    let root = BitMapBackend::new(&filepath, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let ranges = calculate_ranges_2d(data).expect("Should not be empty");
    let mut scatter_ctx = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(ranges.0, ranges.1)?;

    scatter_ctx.configure_mesh().disable_mesh().draw()?;
    scatter_ctx.draw_series(data.axis_iter(Axis(0)).map(|row| {
        let style = map_index_to_color(model.predict(row));
        Circle::new((row[0], row[1]), 2, style)
    }))?;
    root.present()?;

    Ok(())
}

fn calculate_ranges_2d(data: &Array2<f64>) -> Option<(Range<f64>, Range<f64>)> {
    let cmp_fn = |a: &&f64, b: &&f64| a.partial_cmp(b).expect("Should be comparable");

    let x_col = data.column(0);
    let x_min = x_col.iter().min_by(cmp_fn)?.floor();
    let x_max = x_col.iter().max_by(cmp_fn)?.ceil();

    let y_col = data.column(1);
    let y_min = y_col.iter().min_by(cmp_fn)?.floor();
    let y_max = y_col.iter().max_by(cmp_fn)?.ceil();

    let ofs = 2f64;
    Some((x_min - ofs..x_max + ofs, y_min - ofs..y_max + ofs))
}

fn map_index_to_color(index: usize) -> ShapeStyle {
    let color = match index {
        0 => RED,
        1 => GREEN,
        2 => BLUE,
        3 => YELLOW,
        4 => CYAN,
        5 => MAGENTA,
        _ => BLACK,
    };
    color.filled()
}
