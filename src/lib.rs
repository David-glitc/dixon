//! Dixon: a minimal neural network crate for educational purposes: MLP, layers,
//! activations, simple training loops, metrics, and dataset loaders.
//!
//! - MLP with per-sample SGD and CE/MSE losses
//! - Dense layers with configurable activations
//! - Iris and MNIST loaders (CSV/IDX)
//! - Utility helpers for summaries and synthetic data

pub mod activations;
pub mod datasets;
pub mod layers;
pub mod loss;
pub mod metrics;
pub mod network;
pub mod utils;

pub use activations::{Activation, Linear, ReLU, Sigmoid, Softmax};
pub use datasets::{load_iris, load_mnist};
pub use layers::DenseLayer;
pub use loss::{cross_entropy_deriv, cross_entropy_loss, mse_deriv, mse_loss};
pub use metrics::{accuracy, confusion_matrix};
pub use network::{Gradients, MLP};
pub use utils::{generate_synthetic_data, print_model_summary, print_summary_table};
