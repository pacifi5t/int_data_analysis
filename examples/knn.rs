use csv::{ReaderBuilder, StringRecord};
use example_utils::records_into_array;
use int_data_analysis::kmeans::KMeans;
use int_data_analysis::knearest::KNearest;
use ndarray::{s, Array2, Axis};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let clusters = 3;
    let (train, test) = split_train_test(parse_file()?, 0.25);
    let class_labels = clusterize_and_predict(clusters, &train);

    let knn = KNearest::new(3, train, class_labels);
    for each in test.rows() {
        println!("{}", knn.predict(each))
    }

    Ok(())
}

fn parse_file() -> Result<Array2<f64>, csv::Error> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path("data/clusters.csv")?;
    let mut records: Vec<StringRecord> = reader.records().map(|r| r.unwrap()).collect();
    records.shuffle(&mut thread_rng());

    let data = records_into_array(&records);
    Ok(data)
}

fn split_train_test(data: Array2<f64>, test_fraction: f64) -> (Array2<f64>, Array2<f64>) {
    let train_len = ((1.0 - test_fraction) * data.nrows() as f64).ceil() as usize;
    let train = data.slice(s![..train_len, ..]).to_owned();
    let test = data.slice(s![train_len.., ..]).to_owned();
    (train, test)
}

fn clusterize_and_predict(clusters: u32, data: &Array2<f64>) -> Vec<usize> {
    let model = KMeans::default().n_clusters(clusters).fit(data);
    let row_iter = data.axis_iter(Axis(0));
    row_iter.map(|p| model.predict(p) + 1).collect()
}
