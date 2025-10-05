//! Loss functions for training neural networks.
use anyhow::{anyhow, Result};

/// MSE loss
pub fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    if pred.len() != target.len() {
        panic!("Pred and target size mismatch");
    }
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f64>()
        / pred.len() as f64
}

/// MSE deriv
pub fn mse_deriv(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| 2.0 * (p - t) / n)
        .collect()
}

/// Cross-entropy loss (assumes `pred` is a valid probability distribution)
pub fn cross_entropy_loss(pred: &[f64], target: &[f64]) -> Result<f64> {
    if pred.len() != target.len() {
        return Err(anyhow!("Size mismatch"));
    }
    let eps = 1e-12;
    let mut loss = 0.0;
    for (&p, &t) in pred.iter().zip(target) {
        let pp = if p < eps {
            eps
        } else if p > 1.0 - eps {
            1.0 - eps
        } else {
            p
        };
        loss -= t * pp.ln();
    }
    Ok(loss / target.iter().filter(|&&t| t == 1.0).count() as f64)
}

/// CE deriv for softmax + CE: softmax(x) - target
pub fn cross_entropy_deriv(pred: &[f64], target: &[f64]) -> Result<Vec<f64>> {
    if pred.len() != target.len() {
        return Err(anyhow!("Size mismatch"));
    }
    Ok(pred.iter().zip(target).map(|(&p, &t)| p - t).collect())
}
