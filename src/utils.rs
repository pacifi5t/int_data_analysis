use csv::StringRecord;
use ndarray::{Array2, ArrayView1};

pub fn euclidean_distance(point1: ArrayView1<f64>, point2: ArrayView1<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..point1.len() {
        sum += (point1[i].clone() - point2[i].clone()).powi(2);
    }
    sum.sqrt()
}

pub fn records_into_array(records: &Vec<StringRecord>) -> Array2<f64> {
    let shape = (records.len(), records[0].len());
    let vec: Vec<f64> = records
        .iter()
        .flat_map(|rec| rec.iter().map(|str| str.parse::<f64>().unwrap()))
        .collect();
    Array2::from_shape_vec(shape, vec).unwrap()
}
