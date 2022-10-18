use ndarray::{Array2, ArrayView1};
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
        let initial_clusters = self.plus_plus_init(dataset);
        println!("{:?}", initial_clusters);
    }

    fn plus_plus_init(&self, dataset: &Array2<f64>) -> Array2<f64> {
        let mut rng = thread_rng();
        let mut centroid_indexes: Vec<usize> = Vec::new();
        centroid_indexes.push(rng.gen_range(0..dataset.nrows()));

        while centroid_indexes.len() < self.n_clusters as usize {
            let mut max_distance: (usize, f64) = (0, 0.0); // (index of point, distance)
            for i in 0..dataset.nrows() {
                if centroid_indexes.contains(&i) {
                    continue;
                }

                for ci in &centroid_indexes {
                    let distance = Self::euclidian_distance(dataset.row(i), dataset.row(*ci));
                    if distance > max_distance.1 {
                        max_distance = (i, distance);
                    }
                }
            }
            centroid_indexes.push(max_distance.0);
        }

        let mut array: Array2<f64> = Array2::zeros((0, dataset.ncols()));
        for a in centroid_indexes.into_iter().map(|x| dataset.row(x)) {
            array.push_row(a).expect("Row lengths match");
        }
        array
    }

    fn euclidian_distance(point1: ArrayView1<f64>, point2: ArrayView1<f64>) -> f64 {
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
