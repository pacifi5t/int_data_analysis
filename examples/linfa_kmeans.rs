use approx::assert_abs_diff_eq;
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{array, Axis};
use ndarray_rand::rand::SeedableRng;
use polars::prelude::*;
use rand_xoshiro::Xoshiro256Plus;
use std::env::current_dir;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn Error>> {
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let n_clusters = expected_centroids.len_of(Axis(0));

    let filepath = current_dir()?.as_path().join("data/clusters.csv");
    let reader = CsvReader::new(BufReader::new(File::open(filepath)?)).has_header(true);
    let data = reader.finish()?.to_ndarray::<Float64Type>()?;

    let dataset = DatasetBase::from(data.clone());
    let model = KMeans::params_with_rng(n_clusters, Xoshiro256Plus::seed_from_u64(42))
        .tolerance(1e-2)
        .fit(&dataset)
        .expect("KMeans fitted");

    // Once we found our set of centroids, we can also assign new points to the nearest cluster
    let new_observation = DatasetBase::from(array![[-9., 20.5]]);
    // Predict returns the **index** of the nearest cluster
    let dataset = model.predict(new_observation);
    // We can retrieve the actual centroid of the closest cluster using `.centroids()`
    let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
    assert_abs_diff_eq!(
        closest_centroid.to_owned(),
        array![-10., 20.],
        epsilon = 1e-1
    );

    Ok(())
}