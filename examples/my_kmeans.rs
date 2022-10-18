use std::env::current_dir;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use polars::datatypes::Float64Type;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn Error>>{
    let n_clusters = 3;

    let filepath = current_dir()?.as_path().join("data/clusters.csv");
    let reader = CsvReader::new(BufReader::new(File::open(filepath)?)).has_header(true);
    let data = reader.finish()?.to_ndarray::<Float64Type>()?;

    //TODO: Use my own algorithm here

    Ok(())
}