use std::collections::HashMap;
use crate::euclidean_distance;
use ndarray::{Array2, ArrayView1, Axis};

pub struct KNearest {
    k: usize,
    data: Array2<f64>,
    class_labels: Vec<usize>,
}

impl KNearest {
    pub fn new(k: usize, data: Array2<f64>, class_labels: Vec<usize>) -> Self {
        KNearest {
            k,
            data,
            class_labels,
        }
    }

    pub fn predict(&self, point: ArrayView1<f64>) -> usize {
        let mut distances = Vec::new();
        for (i, each) in self.data.axis_iter(Axis(0)).enumerate() {
            let class = self.class_labels[i];
            distances.push((class, euclidean_distance(point, each)))
        }

        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let nearest: Vec<usize> = distances.iter().map(|e| e.0).take(self.k).collect();
        Self::find_dominant_class(&nearest)
    }

    fn find_dominant_class(nearest: &Vec<usize>) -> usize {
        let mut class_counts = HashMap::<usize, usize>::new();
        for each in nearest {
            if let Some(class) = class_counts.get_mut(&each) {
                *class += 1;
            } else {
                class_counts.insert(*each, 1);
            }
        }

        let mut max = (0, 0);
        for (key, value) in class_counts {
            if value > max.1 {
                max = (key, value)
            }
        }
        max.0
    }
}
