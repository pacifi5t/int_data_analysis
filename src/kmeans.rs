use ndarray::{Array1, Array2, Axis, ShapeError};
use rand::{thread_rng, Rng};

pub struct KMeans {
    n_clusters: u32,
    tolerance: f64,
    max_n_iterations: u32,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(3, 1e-2, 100)
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

    pub fn fit(&self, dataset: &Array2<f64>) {
        // let initial_clusters = self.plus_plus_init(dataset);
    }

    fn plus_plus_init(&self, dataset: &Array2<f64>) -> Result<(), ShapeError> {
        let mut rng = thread_rng();
        let mut initial_centroids = Array2::zeros([0, dataset.ndim()]);
        initial_centroids.push(Axis(0), dataset.row(rng.gen()))?;

        Ok(())
    }

    fn euclid_distance(point1: Array1<f64>, point2: Array1<f64>) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..point1.len() {
            sum += (point1[i].clone() - point2[i].clone()).powi(2);
        }
        sum.sqrt()
    }
}

pub struct Model<T> {
    centroids: Array2<T>,
}
