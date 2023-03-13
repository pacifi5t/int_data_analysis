pub use kmeans::{KMeans, Model};
pub use knearest::KNearest;
use ndarray::ArrayView1;

pub mod kmeans;
pub mod knearest;
pub mod example_utils;

pub fn euclidean_distance(point1: ArrayView1<f64>, point2: ArrayView1<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..point1.len() {
        sum += (point1[i] - point2[i]).powi(2);
    }
    sum.sqrt()
}
