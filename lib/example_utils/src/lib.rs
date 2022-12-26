#![allow(unused)]

use csv::StringRecord;
use ndarray::{s, Array1, Array2};
use plotters::prelude::*;
use std::ops::Range;

pub fn records_into_array(records: &Vec<StringRecord>) -> Array2<f64> {
    let shape = (records.len(), records[0].len());
    let vec: Vec<f64> = records
        .iter()
        .flat_map(|rec| rec.iter().map(|str| str.parse::<f64>().unwrap()))
        .collect();
    Array2::from_shape_vec(shape, vec).unwrap()
}

pub fn calculate_ranges_2d(data: &Array2<f64>) -> Option<(Range<f64>, Range<f64>)> {
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

pub fn normalize_data(data: &Array2<f64>) -> Array2<f64> {
    let mut array: Array2<f64> = Array2::zeros((data.nrows(), 0));
    for c in data.columns() {
        let min = c.iter().min_by(|a, b| a.total_cmp(b)).unwrap().to_owned();
        let max = c.iter().max_by(|a, b| a.total_cmp(b)).unwrap().to_owned();
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

pub fn split_train_test(data: Array2<f64>, test_fraction: f64) -> (Array2<f64>, Array2<f64>) {
    let train_len = ((1.0 - test_fraction) * data.nrows() as f64).ceil() as usize;
    let train = data.slice(s![..train_len, ..]).to_owned();
    let test = data.slice(s![train_len.., ..]).to_owned();
    (train, test)
}
