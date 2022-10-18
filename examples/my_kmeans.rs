use csv::{ReaderBuilder, StringRecord};
use int_data_analysis::kmeans::KMeans;
use int_data_analysis::utils::records_into_array;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let n_clusters = 3;

    let filepath = "data/clusters.csv";
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(filepath)?;
    let records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    let data = records_into_array(&records);

    let model = KMeans::default().n_clusters(n_clusters).fit(&data);
    println!("Result {:?}", model.centroids);
    Ok(())
}
