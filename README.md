# primitive_ml

A tiny educational neural network library in Rust. It includes a simple MLP, dense layers, common activations, basic losses/metrics, and dataset loaders for Iris and MNIST. Examples demonstrate training and model persistence.

## Features
- MLP with per-sample SGD and MSE/CE losses
- Dense layers with configurable activations
- Numerically safe softmax + cross-entropy
- Iris CSV and MNIST (IDX gzip) loaders
- Save/load models to `.pere` (gzipped JSON) under `models/`
- Backprop API: compute gradients and apply updates

## Workspace
- Library: `primitive_ml`
- Examples: `ml_examples`

## Quickstart
```bash
# Fetch datasets (MNIST to data/, Iris to example src)
bash scripts/download_datasets.sh

# Run only the Iris example
cargo run -p ml_examples --no-default-features --features iris

# Run only the MNIST example
cargo run -p ml_examples --no-default-features --features mnist
```

## Library usage
```rust
use primitive_ml::{MLP, Sigmoid};
use std::sync::Arc;

let mut mlp = MLP::new(4, vec![10, 5], 3, Arc::new(Sigmoid));
// Train with built-in loop or your own

// Save / load
mlp.save_pere("models/iris_model.pere")?;
let reloaded = MLP::load_pere("models/iris_model.pere")?;
```

## Backprop API (custom training)
```rust
use primitive_ml::Gradients;
let grads: Gradients = mlp.compute_gradients(&input, &target, "ce")?;
mlp.apply_gradients(&grads, 0.05);
```

## Notes
- This is educational code: single-threaded, per-sample SGD.
- MNIST loader looks in `data/` first, then current directory.

## License
MIT
