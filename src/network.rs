// primitive_ml/src/network.rs
use crate::activations::{Activation, Softmax, ActivationKind, identify_activation_kind};
use std::sync::Arc;
use crate::{cross_entropy_loss, mse_loss};
use crate::layers::DenseLayer;
use crate::loss::mse_deriv;
use crate::metrics::accuracy;
use anyhow::{anyhow, Result};
use std::fmt;
use serde::{Serialize, Deserialize};
use flate2::write::GzEncoder;
use flate2::Compression;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Read, Write};

/// MLP
#[derive(Debug)]
pub struct MLP {
    /// Ordered list of dense layers from input to output.
    pub layers: Vec<DenseLayer>,
    /// Number of input features.
    input_size: usize,
    /// Number of outputs/classes.
    output_size: usize,
}

impl MLP {
    /// Create a new MLP with the given sizes.
    ///
    /// - `input_size`: number of input features
    /// - `hidden_sizes`: sizes of hidden layers, in order
    /// - `output_size`: number of outputs/classes
    /// - `activation`: activation used for all layers
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        activation: Arc<dyn Activation + Send + Sync>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;
        for &size in &hidden_sizes {
            layers.push(DenseLayer::new(prev_size, size, activation.clone()));
            prev_size = size;
        }
        // Output layer
        layers.push(DenseLayer::new(prev_size, output_size, activation));
        Self {
            layers,
            input_size,
            output_size,
        }
    }

    /// Forward pass from input to output.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let (_, a) = layer.forward(&current);
            current = a;
        }
        current
    }

    /// Train with MSE or cross-entropy (loss_type: "mse" or "ce").
    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)], epochs: usize, lr: f64, loss_type: &str) -> Result<()> {
        if dataset.is_empty() {
            return Err(anyhow!("Dataset is empty"));
        }
        let mut losses = Vec::new();
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in dataset {
                if input.len() != self.input_size || target.len() != self.output_size {
                    return Err(anyhow!("Input/target size mismatch"));
                }
                // Forward
                let mut activations = vec![input.clone()];
                let mut zs = Vec::new();
                let mut current = input.clone();
                for layer in &self.layers {
                    let (z, a) = layer.forward(&current);
                    zs.push(z);
                    activations.push(a.clone());
                    current = a;
                }
                // For CE, use softmax over final logits
                let logits_last = zs.last().expect("No layers in MLP");
                let (pred, loss) = if loss_type == "ce" {
                    let mut y_hat = Softmax.apply_vec(logits_last);
                    // Guard against denormals / NaNs
                    let eps = 1e-12;
                    for p in &mut y_hat {
                        if !p.is_finite() || *p < eps {
                            *p = eps;
                        } else if *p > 1.0 - eps {
                            *p = 1.0 - eps;
                        }
                    }
                    let l = cross_entropy_loss(&y_hat, target)?;
                    (y_hat, l)
                } else {
                    let l = mse_loss(&current, target);
                    (current, l)
                };
                total_loss += loss;
                // Backward
                let mut delta = if loss_type == "ce" {
                    // softmax + CE: dz_last = y_hat - target
                    pred.iter().zip(target).map(|(&p, &t)| p - t).collect()
                } else {
                    mse_deriv(&pred, target)
                };
                let last_layer_index = self.layers.len() - 1;
                for layer_idx in (0..self.layers.len()).rev() {
                    let layer = &mut self.layers[layer_idx];
                    let z = &zs[layer_idx];
                    let a_prev = &activations[layer_idx];
                    // For CE, skip activation derivative at the output layer
                    let dz: Vec<f64> = if loss_type == "ce" && layer_idx == last_layer_index {
                        delta.clone()
                    } else {
                        delta.iter().zip(z).map(|(&d, &val)| d * layer.activation.derivative(val)).collect()
                    };
                    // Update
                    layer.update(a_prev, &dz, lr);
                    // Propagate delta = W^T * dz
                    delta = vec![0.0; a_prev.len()];
                    for (i, row) in layer.weights.iter().enumerate() {
                        for (j, &w) in row.iter().enumerate() {
                            delta[j] += w * dz[i];
                        }
                    }
                }
            }
            let avg_loss = total_loss / dataset.len() as f64;
            losses.push(avg_loss);
            println!("Epoch {}: Loss = {:.6}", epoch + 1, avg_loss);
        }
        // Print summary table
        crate::utils::print_summary_table(&losses, "Training Loss");
        Ok(())
    }

    /// Predict outputs for a single input.
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input)
    }

    /// Evaluate accuracy assuming one-hot targets.
    pub fn evaluate(&self, dataset: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        accuracy(dataset, self)
    }

    /// Save model to .pere (gzipped JSON).
    pub fn save_pere(&self, path: &str) -> Result<()> {
        let dto = MlpDto::from_mlp(self);
        let json = serde_json::to_vec(&dto)?;
        if let Some(parent) = std::path::Path::new(path).parent() { std::fs::create_dir_all(parent)?; }
        let file = File::create(path)?;
        let mut enc = GzEncoder::new(file, Compression::default());
        enc.write_all(&json)?;
        enc.finish()?;
        Ok(())
    }

    /// Load model from .pere (gzipped JSON)
    pub fn load_pere(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mut dec = GzDecoder::new(file);
        let mut buf = Vec::new();
        dec.read_to_end(&mut buf)?;
        let dto: MlpDto = serde_json::from_slice(&buf)?;
        Ok(dto.into_mlp())
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sizes = vec![self.input_size];
        for layer in &self.layers {
            sizes.push(layer.bias.len());
        }
        write!(f, "MLP: {:?}", sizes)
    }
}

// ============ Persistence DTOs ============

#[derive(Debug, Serialize, Deserialize)]
struct LayerDto {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f64>>, // [output_size][input_size]
    bias: Vec<f64>,         // [output_size]
    activation: ActivationKind,
}

#[derive(Debug, Serialize, Deserialize)]
struct MlpDto {
    input_size: usize,
    output_size: usize,
    layers: Vec<LayerDto>,
}

impl MlpDto {
    fn from_mlp(mlp: &MLP) -> Self {
        let mut prev = mlp.input_size;
        let layers = mlp
            .layers
            .iter()
            .map(|layer| {
                let act_kind = identify_activation_kind(layer.activation.as_ref());
                let out = layer.bias.len();
                let dto = LayerDto {
                    input_size: prev,
                    output_size: out,
                    weights: layer.weights.clone(),
                    bias: layer.bias.clone(),
                    activation: act_kind,
                };
                prev = out;
                dto
            })
            .collect();
        Self {
            input_size: mlp.input_size,
            output_size: mlp.output_size,
            layers,
        }
    }

    fn into_mlp(self) -> MLP {
        use std::sync::Arc;
        let mut layers: Vec<DenseLayer> = Vec::with_capacity(self.layers.len());
        for ld in &self.layers {
            let mut layer = DenseLayer::new(
                ld.input_size,
                ld.output_size,
                ld.activation.to_arc(),
            );
            layer.weights = ld.weights.clone();
            layer.bias = ld.bias.clone();
            layers.push(layer);
        }
        MLP {
            layers,
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }
}