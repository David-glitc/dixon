//! A minimal neural network crate for educational purposes: MLP, layers,
//! activations, simple training loops, metrics, and dataset loaders.
//!
//! - MLP with per-sample SGD and CE/MSE losses
//! - Dense layers with configurable activations
//! - Iris and MNIST loaders (CSV/IDX)
//! - Utility helpers for summaries and synthetic data

pub mod activations;
pub mod layers;
pub mod network;
pub mod loss;
pub mod metrics;
pub mod datasets;
pub mod utils;

pub use activations::{Activation, ReLU, Sigmoid, Softmax, Linear};
pub use layers::DenseLayer;
pub use network::MLP;
pub use loss::{mse_loss, mse_deriv, cross_entropy_loss, cross_entropy_deriv};
pub use metrics::{accuracy, confusion_matrix};
pub use datasets::{load_iris, load_mnist};
pub use utils::{print_model_summary, generate_synthetic_data, print_summary_table};