//! Utility functions for neural network training and evaluation.
use crate::network::MLP;
use rand::Rng;

/// Generate synthetic data
pub fn generate_synthetic_data(n_samples: usize, input_size: usize, output_size: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = rand::thread_rng();
    (0..n_samples).map(|_| {
        let input: Vec<f64> = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let target: Vec<f64> = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        (input, target)
    }).collect()
}

/// Print model summary
pub fn print_model_summary(mlp: &MLP) {
    println!("Model Summary:\n{}", mlp);
}

/// Print simple table for losses
pub fn print_summary_table(values: &[f64], title: &str) {
    println!("\n{} Summary Table:", title);
    println!("+----------------+----------+");
    println!("| Epoch Range   | Avg Value|");
    println!("+----------------+----------+");
    if !values.is_empty() {
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        println!("| All Epochs    | {:>8.6} |", avg);
    }
    println!("+----------------+----------+");
}