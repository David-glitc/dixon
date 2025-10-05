//! Dataset loading utilities for Iris and MNIST.
use anyhow::{anyhow, Result};
use byteorder::{BigEndian, ReadBytesExt};
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};

pub type Dataset = Vec<(Vec<f64>, Vec<f64>)>;

/// One-hot encode
pub fn one_hot(label: usize, num_classes: usize) -> Vec<f64> {
    let mut v = vec![0.0; num_classes];
    if label < num_classes {
        v[label] = 1.0;
    }
    v
}

/// Load Iris from CSV
pub fn load_iris(filename: &str) -> Result<Dataset> {
    let file = File::open(filename).map_err(|e| anyhow!("Failed to open {}: {}", filename, e))?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut dataset = Vec::new();
    let mut species_map = HashMap::new();
    species_map.insert("setosa".to_string(), 0);
    species_map.insert("versicolor".to_string(), 1);
    species_map.insert("virginica".to_string(), 2);

    for result in rdr.records() {
        let record = result.map_err(|e| anyhow!("CSV parse error: {}", e))?;
        if record.len() != 5 {
            continue;
        }
        let features: Vec<f64> = record
            .iter()
            .take(4)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        let species = record[4].trim_matches('"').to_lowercase();
        // Normalize values like "Iris-setosa" -> "setosa"
        let species_norm = species.trim().trim_start_matches("iris-").to_string();
        let label = *species_map
            .get(&species_norm)
            .ok_or_else(|| anyhow!("Unknown species: {}", species))?;
        let target = one_hot(label, 3);
        dataset.push((features, target));
    }
    if dataset.is_empty() {
        return Err(anyhow!("No data loaded from Iris"));
    }
    Ok(dataset)
}

/// MNIST loader
#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(filename: &str) -> Result<Self> {
        let file =
            File::open(filename).map_err(|e| anyhow!("Failed to open {}: {}", filename, e))?;
        let mut gz = GzDecoder::new(file);
        let mut contents = Vec::new();
        gz.read_to_end(&mut contents)
            .map_err(|e| anyhow!("Gzip read error: {}", e))?;
        let mut r = Cursor::new(&contents);
        let magic = r
            .read_i32::<BigEndian>()
            .map_err(|e| anyhow!("Read magic: {}", e))?;
        let mut sizes = Vec::new();
        let mut data = Vec::new();
        match magic {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => return Err(anyhow!("Invalid magic: {}", magic)),
        }
        r.read_to_end(&mut data)
            .map_err(|e| anyhow!("Read data: {}", e))?;
        Ok(Self { sizes, data })
    }
}

/// Load MNIST train/test
pub fn load_mnist(train: bool) -> Result<Dataset> {
    let prefix = if train { "train" } else { "t10k" };
    let label_name = format!("{}-labels-idx1-ubyte.gz", prefix);
    let image_name = format!("{}-images-idx3-ubyte.gz", prefix);
    let label_path = format!("data/{}", label_name);
    let image_path = format!("data/{}", image_name);
    let label_data = MnistData::new(&label_path).or_else(|_| MnistData::new(&label_name))?;
    let image_data = MnistData::new(&image_path).or_else(|_| MnistData::new(&image_name))?;
    let num_images = label_data.sizes[0] as usize;
    let image_size = 28 * 28;
    let mut dataset = Vec::new();
    for i in 0..num_images {
        let start = i * image_size;
        if start + image_size > image_data.data.len() {
            return Err(anyhow!("Image data overflow"));
        }
        let img_bytes = &image_data.data[start..start + image_size];
        let input: Vec<f64> = img_bytes.iter().map(|&b| b as f64 / 255.0).collect();
        let label = label_data.data[i] as usize;
        let target = one_hot(label, 10);
        dataset.push((input, target));
    }
    if dataset.is_empty() {
        return Err(anyhow!("No MNIST data loaded"));
    }
    Ok(dataset)
}
