//! Dense layer implementation with weights, bias, and activation function.
use crate::activations::Activation;
use std::sync::Arc;
use rand::Rng;

/// Matrix type
pub type Matrix = Vec<Vec<f64>>;

/// A fully-connected (dense) layer with weights, bias, and an activation function.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Matrix,
    pub bias: Vec<f64>,
    pub activation: Arc<dyn Activation + Send + Sync>,
}

impl DenseLayer {
    /// Create a new dense layer using He (Kaiming) uniform initialization and small positive bias.
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Arc<dyn Activation + Send + Sync>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        // He uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
        let limit = (6.0f64 / (input_size as f64)).sqrt();
        let weights: Matrix = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-limit..limit)).collect())
            .collect();
        let bias = vec![0.01; output_size];
        Self { weights, bias, activation }
    }

    /// Forward pass: computes pre-activations `z = W·x + b` and activations `a = act(z)`.
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let z: Vec<f64> = self.weights.iter().zip(&self.bias).map(|(row, &b)| {
            row.iter().zip(input).map(|(&w, &i)| w * i).sum::<f64>() + b
        }).collect();
        let a: Vec<f64> = z.iter().map(|&val| self.activation.apply(val)).collect();
        (z, a)
    }

    /// Backward helper: given `dL/da` and `z`, compute `dL/da_prev`.
    pub fn backward(&self, da: &[f64], z: &[f64]) -> Vec<f64> {
        // dz = da * act'(z)
        let dz: Vec<f64> = da.iter().zip(z).map(|(&d, &val)| d * self.activation.derivative(val)).collect();
        // da_prev = W^T * dz
        let mut da_prev = vec![0.0; self.weights[0].len()];
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                da_prev[j] += w * dz[i];
            }
        }
        da_prev
    }

    /// Parameter update: `W -= lr * (dz ⊗ input)`, `b -= lr * dz`.
    pub fn update(&mut self, input: &[f64], dz: &[f64], lr: f64) {
        // bias
        for (b, &d) in self.bias.iter_mut().zip(dz) {
            *b -= lr * d;
        }
        // weights
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w -= lr * dz[i] * input[j];
            }
        }
    }
}