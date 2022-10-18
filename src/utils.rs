use csv::StringRecord;
use ndarray::Array2;

pub fn records_into_array(records: &Vec<StringRecord>) -> Array2<f64> {
    let shape = (records.len(), records[0].len());
    let vec: Vec<f64> = records
        .iter()
        .flat_map(|rec| rec.iter().map(|str| str.parse::<f64>().unwrap()))
        .collect();
    Array2::from_shape_vec(shape, vec).unwrap()
}
