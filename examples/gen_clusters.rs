use csv::Writer;
use linfa_datasets::generate;
use ndarray::array;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;
use std::error::Error;
use std::fs::create_dir;

fn main() -> Result<(), Box<dyn Error>> {
    let expected_centroids = array![[0., 1.], [-10., 20.], [-1., 10.]];
    let data = generate::blobs(
        100,
        &expected_centroids,
        &mut Xoshiro256Plus::seed_from_u64(42),
    );

    create_dir("data").unwrap_or(());
    let mut writer = Writer::from_path("data/clusters.csv")?;

    writer.write_record(["x", "y"])?;
    for row in data.rows() {
        let vec: Vec<String> = row.into_iter().map(|e| e.to_string()).collect();
        writer.write_record(vec.as_slice())?;
    }

    Ok(())
}
