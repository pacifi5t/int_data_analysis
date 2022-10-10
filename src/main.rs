use approx::assert_abs_diff_eq;
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_datasets::generate;
use ndarray::{array, Axis};
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

fn main() {
    // Our random number generator, seeded for reproducibility
    let seed = 42;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    // `expected_centroids` has shape `(n_centroids, n_features)`
    // i.e. three points in the 2-dimensional plane
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    // Let's generate a synthetic dataset: three blobs of observations
    // (100 points each) centered around our `expected_centroids`
    let data = generate::blobs(100, &expected_centroids, &mut rng);
    let n_clusters = expected_centroids.len_of(Axis(0));

    let observations = DatasetBase::from(data.clone());
    // Let's configure and run our K-means algorithm
    // We use the builder pattern to specify the hyperparameters
    // `n_clusters` is the only mandatory parameter.
    // If you don't specify the others (e.g. `n_runs`, `tolerance`, `max_n_iterations`)
    // default values will be used.
    let model = KMeans::params_with_rng(n_clusters, rng.clone())
        .tolerance(1e-2)
        .fit(&observations)
        .expect("KMeans fitted");

    // Once we found our set of centroids, we can also assign new points to the nearest cluster
    let new_observation = DatasetBase::from(array![[-9., 20.5]]);
    // Predict returns the **index** of the nearest cluster
    let dataset = model.predict(new_observation);
    // We can retrieve the actual centroid of the closest cluster using `.centroids()`
    let closest_centroid = &model.centroids().index_axis(Axis(0), dataset.targets()[0]);
    let owned_centroid = closest_centroid.to_owned();
    assert_abs_diff_eq!(owned_centroid, array![-10., 20.], epsilon = 1e-1);
}
