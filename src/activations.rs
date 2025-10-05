use std::fmt;
use std::any::Any;
use serde::{Serialize, Deserialize};

/// Trait for activation functions.
pub trait Activation: fmt::Debug + Send + Sync + Any {
    fn apply(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
    fn apply_vec(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.apply(xi)).collect()
    }
}

/// ReLU: max(0, x)
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl Activation for ReLU {
    fn apply(&self, x: f64) -> f64 {
        x.max(0.0)
    }
    fn derivative(&self, x: f64) -> f64 {
        (x > 0.0) as u8 as f64
    }
}

/// Sigmoid: 1 / (1 + exp(-x))
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        let s = self.apply(x);
        s * (1.0 - s)
    }
}

/// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl Activation for Tanh {
    fn apply(&self, x: f64) -> f64 {
        x.tanh()
    }
    fn derivative(&self, x: f64) -> f64 {
        let t = self.apply(x);
        1.0 - t * t
    }
}

/// LeakyReLU: x if x > 0 else alpha * x (alpha=0.01 default)
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    pub alpha: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl Activation for LeakyReLU {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.alpha * x }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { self.alpha }
    }
}

/// ELU: x if x > 0 else alpha*(exp(x)-1) (alpha=1.0 default)
#[derive(Debug, Clone)]
pub struct ELU {
    pub alpha: f64,
}

impl Default for ELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Activation for ELU {
    fn apply(&self, x: f64) -> f64 {
        if x > 0.0 { x } else { self.alpha * (x.exp() - 1.0) }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { self.apply(x) + self.alpha }
    }
}

/// Swish (SiLU): x * sigmoid(beta * x) (beta=1.0 default)
#[derive(Debug, Clone)]
pub struct Swish {
    pub beta: f64,
}

impl Default for Swish {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl Activation for Swish {
    fn apply(&self, x: f64) -> f64 {
        // Sigmoid is a unit struct; call its method on the value
        x * Sigmoid.apply(self.beta * x)
    }
    fn derivative(&self, x: f64) -> f64 {
        let s = Sigmoid.apply(self.beta * x);
        let ds = s * (1.0 - s) * self.beta;
        s + x * ds
    }
}

/// Softmax (vector-only)
#[derive(Debug, Clone, Default)]
pub struct Softmax;

impl Activation for Softmax {
    fn apply(&self, _x: f64) -> f64 {
        unimplemented!("Softmax is vector-only; use apply_vec")
    }
    fn derivative(&self, _x: f64) -> f64 {
        unimplemented!()
    }
}

impl Softmax {
    pub fn apply_vec(&self, x: &[f64]) -> Vec<f64> {
        if x.is_empty() {
            return Vec::new();
        }
        let max = x.iter().fold(f64::MIN, |a, &b| a.max(b));
        let exps: Vec<f64> = x.iter().map(|&xi| (xi - max).exp()).collect();
        let exp_sum: f64 = exps.iter().sum();
        if !exp_sum.is_finite() || exp_sum <= 0.0 {
            // Fallback to uniform distribution to avoid NaNs
            let n = x.len() as f64;
            return vec![1.0 / n; x.len()];
        }
        exps.into_iter().map(|e| e / exp_sum).collect()
    }

    pub fn derivative_vec(&self, y: &[f64], i: usize, j: usize) -> f64 {
        if i == j {
            y[i] * (1.0 - y[i])
        } else {
            -y[i] * y[j]
        }
    }
}

/// Linear: identity
#[derive(Debug, Clone, Default)]
pub struct Linear;

impl Activation for Linear {
    fn apply(&self, x: f64) -> f64 {
        x
    }
    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}

/// Serializable activation kinds for persistence
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationKind {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Swish,
    Softmax,
    Linear,
}

impl ActivationKind {
    pub fn to_arc(&self) -> std::sync::Arc<dyn Activation + Send + Sync> {
        use std::sync::Arc;
        match self {
            ActivationKind::ReLU => Arc::new(ReLU),
            ActivationKind::Sigmoid => Arc::new(Sigmoid),
            ActivationKind::Tanh => Arc::new(Tanh),
            ActivationKind::LeakyReLU => Arc::new(LeakyReLU::default()),
            ActivationKind::ELU => Arc::new(ELU::default()),
            ActivationKind::Swish => Arc::new(Swish::default()),
            ActivationKind::Softmax => Arc::new(Softmax),
            ActivationKind::Linear => Arc::new(Linear),
        }
    }
}

/// Best-effort identification of activation kind from a trait object
pub fn identify_activation_kind(a: &(dyn Activation + Send + Sync)) -> ActivationKind {
    let any = a as &dyn Any;
    if any.is::<ReLU>() { return ActivationKind::ReLU; }
    if any.is::<Sigmoid>() { return ActivationKind::Sigmoid; }
    if any.is::<Tanh>() { return ActivationKind::Tanh; }
    if any.is::<LeakyReLU>() { return ActivationKind::LeakyReLU; }
    if any.is::<ELU>() { return ActivationKind::ELU; }
    if any.is::<Swish>() { return ActivationKind::Swish; }
    if any.is::<Softmax>() { return ActivationKind::Softmax; }
    ActivationKind::Linear
}