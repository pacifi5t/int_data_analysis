pub struct KMeansParameters {
    tolerance: f64,
    max_n_iterations: u32,
}

impl KMeansParameters {
    pub fn new(tolerance: f64, max_n_iterations: u32) -> Self {
        KMeansParameters {
            tolerance,
            max_n_iterations,
        }
    }

    pub fn tolerance(&mut self, value: f64) -> &mut KMeansParameters {
        self.tolerance = value;
        self
    }

    pub fn max_n_iterations(&mut self, value: u32) -> &mut KMeansParameters {
        self.max_n_iterations = value;
        self
    }
}

impl Default for KMeansParameters {
    fn default() -> Self {
        Self::new(1e-2, 100)
    }
}

pub struct KMeans {
    params: KMeansParameters,
}

impl KMeans {
    pub fn new(params: KMeansParameters) -> Self {
        KMeans { params }
    }
}
