// ml_examples/src/main.rs
use anyhow::Result;
use dixon::{load_iris, load_mnist, print_model_summary, Linear, ReLU, Sigmoid, MLP};
use std::sync::Arc;

fn main() -> Result<()> {
    #[cfg(feature = "iris")]
    {
        println!("=== Iris Dataset ===");
        let iris_csv = "data/iris.csv";
        let iris_data = load_iris(iris_csv)?;
        let mut mlp_iris = MLP::new(4, vec![10, 5], 3, Arc::new(Sigmoid));
        print_model_summary(&mlp_iris);
        mlp_iris.train(&iris_data, 1000, 0.1, "ce")?;
        let iris_acc = mlp_iris.evaluate(&iris_data);
        println!("Iris Accuracy: {:.2}%", iris_acc * 100.0);

        // Demo: save and load model
        mlp_iris.save_pere("models/iris_model.pere")?;
        let reloaded_iris = MLP::load_pere("models/iris_model.pere")?;
        let iris_acc_loaded = reloaded_iris.evaluate(&iris_data);
        println!("Iris Accuracy (reloaded): {:.2}%", iris_acc_loaded * 100.0);

        // Backprop demo: manual SGD (small epochs for illustration)
        println!("\n--- Iris Backprop Demo (manual SGD) ---");
        let mut iris_bp = MLP::new(4, vec![10, 5], 3, Arc::new(Sigmoid));
        let epochs_bp = 20;
        let lr_bp = 0.1;
        for _ in 0..epochs_bp {
            let mut loss_sum = 0.0;
            for (x, y) in &iris_data {
                let grads = iris_bp.compute_gradients(x, y, "ce")?;
                iris_bp.apply_gradients(&grads, lr_bp);
                // reuse forward output via predict for quick CE approx (not exact batch loss)
                let pred = iris_bp.predict(x);
                loss_sum += dixon::cross_entropy_loss(&pred, y)?;
            }
            let avg = loss_sum / iris_data.len() as f64;
            println!("Iris BP Avg Loss: {:.6}", avg);
        }
        println!(
            "Iris BP Accuracy: {:.2}%",
            iris_bp.evaluate(&iris_data) * 100.0
        );
    }

    #[cfg(feature = "mnist")]
    {
        println!("\n=== MNIST Subset (first 1000) ===");
        let mut mnist_data = load_mnist(true)?;
        mnist_data.truncate(1000);
        let mut mlp_mnist = MLP::new(784, vec![128, 64], 10, Arc::new(ReLU));
        // Use Linear activation on output to align with softmax-on-logits in loss
        mlp_mnist
            .layers
            .last_mut()
            .expect("output layer")
            .activation = Arc::new(Linear);
        print_model_summary(&mlp_mnist);
        mlp_mnist.train(&mnist_data, 20, 0.05, "ce")?;
        let mnist_acc = mlp_mnist.evaluate(&mnist_data);
        println!("MNIST Accuracy: {:.2}%", mnist_acc * 100.0);

        // Demo: save and load model
        mlp_mnist.save_pere("models/mnist_model.pere")?;
        match MLP::load_pere("models/mnist_model.pere") {
            Ok(reloaded_mnist) => {
                let mnist_acc_loaded = reloaded_mnist.evaluate(&mnist_data);
                println!(
                    "MNIST Accuracy (reloaded): {:.2}%",
                    mnist_acc_loaded * 100.0
                );
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load mnist_model.pere ({}). Skipping load demo.",
                    e
                );
            }
        }

        // Backprop demo: manual SGD on a small subset for speed
        println!("\n--- MNIST Backprop Demo (manual SGD, 200 samples) ---");
        let mut mnist_bp = MLP::new(784, vec![128, 64], 10, Arc::new(ReLU));
        let mut subset = mnist_data.clone();
        subset.truncate(200);
        let epochs_bp = 3;
        let lr_bp = 0.03;
        for _ in 0..epochs_bp {
            let mut loss_sum = 0.0;
            for (x, y) in &subset {
                let grads = mnist_bp.compute_gradients(x, y, "ce")?;
                mnist_bp.apply_gradients(&grads, lr_bp);
                let pred = mnist_bp.predict(x);
                loss_sum += dixon::cross_entropy_loss(&pred, y)?;
            }
            let avg = loss_sum / subset.len() as f64;
            println!("MNIST BP Avg Loss: {:.6}", avg);
        }
        println!(
            "MNIST BP (subset) Accuracy: {:.2}%",
            mnist_bp.evaluate(&subset) * 100.0
        );
    }

    Ok(())
}
