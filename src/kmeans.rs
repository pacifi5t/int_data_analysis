use crate::utils::euclidean_distance;
use approx::abs_diff_eq;
use ndarray::{Array2, ArrayView1, Axis};
use ndarray_rand::rand_distr::num_traits::Float;
use rand::{thread_rng, Rng};

pub struct KMeans {
    n_clusters: u32,
    tolerance: f64,
    max_n_iterations: u32,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(3, 1e-1, 100)
    }
}

impl KMeans {
    pub fn new(n_clusters: u32, tolerance: f64, max_n_iterations: u32) -> Self {
        KMeans {
            n_clusters,
            tolerance,
            max_n_iterations,
        }
    }

    pub fn n_clusters(&mut self, value: u32) -> &mut Self {
        self.n_clusters = value;
        self
    }

    pub fn tolerance(&mut self, value: f64) -> &mut Self {
        self.tolerance = value;
        self
    }

    pub fn max_n_iterations(&mut self, value: u32) -> &mut Self {
        self.max_n_iterations = value;
        self
    }

    pub fn fit(&self, dataset: &Array2<f64>) -> Model {
        let mut centroids = self.plus_plus_init(dataset);
        for _ in 0..self.max_n_iterations {
            let previous_centroids = centroids.clone();
            let mut clustered_data_indexes = self.init_clustered_data_index_vec();

            for ei in 0..dataset.nrows() {
                let closest_centroid = Self::get_closest_centroid(dataset.row(ei), &centroids);
                clustered_data_indexes[closest_centroid.0].push(ei);
            }

            for ci in 0..centroids.nrows() {
                let mut array: Array2<f64> = Array2::zeros((0, centroids.ncols()));
                for ei in &clustered_data_indexes[ci] {
                    array.push_row(dataset.row(*ei)).unwrap();
                }
                let new_centroid = array.mean_axis(Axis(0)).expect("Should be > 0");
                centroids.row_mut(ci).assign(&new_centroid);
            }

            if abs_diff_eq!(centroids, previous_centroids, epsilon = self.tolerance) {
                break;
            }
        }
        Model { centroids }
    }

    fn plus_plus_init(&self, dataset: &Array2<f64>) -> Array2<f64> {
        let mut centroid_indexes: Vec<usize> = Vec::new();
        centroid_indexes.push(thread_rng().gen_range(0..dataset.nrows()));

        while centroid_indexes.len() < self.n_clusters as usize {
            let mut max_distance: (usize, f64) = (0, 0.0); // (index of point, distance)
            for i in 0..dataset.nrows() {
                if centroid_indexes.contains(&i) {
                    continue;
                }

                for ci in &centroid_indexes {
                    let distance = euclidean_distance(dataset.row(i), dataset.row(*ci));
                    if distance > max_distance.1 {
                        max_distance = (i, distance);
                    }
                }
            }
            centroid_indexes.push(max_distance.0);
        }

        let mut array: Array2<f64> = Array2::zeros((0, dataset.ncols()));
        for a in centroid_indexes.into_iter().map(|i| dataset.row(i)) {
            array.push_row(a).expect("Row lengths match");
        }
        array
    }

    fn init_clustered_data_index_vec(&self) -> Vec<Vec<usize>> {
        let mut clustered_data_indexes: Vec<Vec<usize>> = Vec::new();
        for _ in 0..self.n_clusters {
            clustered_data_indexes.push(Vec::new());
        }
        clustered_data_indexes
    }

    fn get_closest_centroid(point: ArrayView1<f64>, centroids: &Array2<f64>) -> (usize, f64) {
        let mut closest_centroid = (0, f64::max_value());
        for ci in 0..centroids.nrows() {
            let distance = euclidean_distance(point, centroids.row(ci));
            if distance < closest_centroid.1 {
                closest_centroid = (ci, distance);
            }
        }
        closest_centroid
    }
}

pub struct Model {
    centroids: Array2<f64>,
}

impl Model {
    pub fn centroids(&self) -> Array2<f64> {
        self.centroids.clone()
    }
}
