use linfa_datasets::generate;
use ndarray::{array, ArrayView1};
use ndarray_rand::rand::SeedableRng;
use polars::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use std::env::current_dir;
use std::error::Error;
use std::fs::File;

fn main() -> Result<(), Box<dyn Error>> {
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let data = generate::blobs(
        100,
        &expected_centroids,
        &mut Xoshiro256Plus::seed_from_u64(42),
    );
    let mut df = DataFrame::new(vec![
        column_into_series("x", &data.column(0)),
        column_into_series("y", &data.column(1)),
    ])?;
    let filepath = current_dir()?.as_path().join("data/clusters.csv");
    CsvWriter::new(File::create(filepath)?)
        .has_header(true)
        .finish(&mut df)?;

    Ok(())
}

fn column_into_series(name: &str, data: &ArrayView1<f64>) -> Series {
    Float64Chunked::new(name, data.into_owned().into_raw_vec()).into_series()
}
