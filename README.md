# Dixon

An educational neural network library in Rust. It includes a simple MLP, dense layers, common activations, basic losses/metrics, and dataset loaders for Iris and MNIST. Examples demonstrate training, manual backprop, and model persistence.

## Features
- MLP with per-sample SGD and MSE/CE losses
- Dense layers with configurable activations
- Numerically safe softmax + cross-entropy
- Iris CSV and MNIST (IDX gzip) loaders
- Save/load models to `.pere` (gzipped JSON) under `models/`
- Backprop API: compute gradients and apply updates

## Workspace
- Library: `dixon`
- Examples: `ml_examples`

## Quickstart
```bash
# Fetch datasets (MNIST to data/, Iris to data/)
bash scripts/download_datasets.sh

# Run only the Iris example
cargo run -p ml_examples --no-default-features --features iris

# Run only the MNIST example
cargo run -p ml_examples --no-default-features --features mnist
```

## Library usage (basic)
```rust
use dixon::{MLP, Sigmoid};
use std::sync::Arc;

let mut mlp = MLP::new(4, vec![10, 5], 3, Arc::new(Sigmoid));
// Train with built-in loop or your own

// Save / load
mlp.save_pere("models/iris_model.pere")?;
let reloaded = MLP::load_pere("models/iris_model.pere")?;
```

## Backprop API (custom training)
```rust
use dixon::Gradients;
let grads: Gradients = mlp.compute_gradients(&input, &target, "ce")?;
mlp.apply_gradients(&grads, 0.05);
```

## Notes
- This is educational code: single-threaded, per-sample SGD. For best MNIST results, shuffle per-epoch (enabled), consider Linear output with CE (enabled in example), lower LR (e.g., 0.01), and run more epochs.
- MNIST loader looks in `data/` first, then current directory.

## Docs (GitHub Pages)
When pushed to `master`/`main`, API docs are published to GitHub Pages from `gh-pages`. Enable in repository Settings → Pages → Branch: `gh-pages`.

## License
MIT
