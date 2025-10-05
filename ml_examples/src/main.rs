// ml_examples/src/main.rs
use anyhow::Result;
use primitive_ml::{
    load_iris, load_mnist, print_model_summary, print_summary_table, ReLU, Sigmoid, MLP,
};
use std::sync::Arc;

fn main() -> Result<()> {
    #[cfg(feature = "iris")]
    {
        println!("=== Iris Dataset ===");
        let iris_csv = concat!(env!("CARGO_MANIFEST_DIR"), "/src/iris.csv");
        let iris_data = load_iris(iris_csv)?;
        let mut mlp_iris = MLP::new(4, vec![10, 5], 3, Arc::new(Sigmoid));
        print_model_summary(&mlp_iris);
        mlp_iris.train(&iris_data, 200, 0.1, "ce")?;
        let iris_acc = mlp_iris.evaluate(&iris_data);
        println!("Iris Accuracy: {:.2}%", iris_acc * 100.0);

        // Demo: save and load model
        mlp_iris.save_pere("models/iris_model.pere")?;
        let reloaded_iris = MLP::load_pere("models/iris_model.pere")?;
        let iris_acc_loaded = reloaded_iris.evaluate(&iris_data);
        println!("Iris Accuracy (reloaded): {:.2}%", iris_acc_loaded * 100.0);
    }

    #[cfg(feature = "mnist")]
    {
        println!("\n=== MNIST Subset (first 1000) ===");
        let mut mnist_data = load_mnist(true)?;
        mnist_data.truncate(1000);
        let mut mlp_mnist = MLP::new(784, vec![128, 64], 10, Arc::new(ReLU));
        print_model_summary(&mlp_mnist);
        mlp_mnist.train(&mnist_data, 20, 0.05, "ce")?;
        let mnist_acc = mlp_mnist.evaluate(&mnist_data);
        println!("MNIST Accuracy: {:.2}%", mnist_acc * 100.0);

        // Demo: save and load model
        mlp_mnist.save_pere("models/mnist_model.pere")?;
        let reloaded_mnist = MLP::load_pere("models/mnist_model.pere")?;
        let mnist_acc_loaded = reloaded_mnist.evaluate(&mnist_data);
        println!("MNIST Accuracy (reloaded): {:.2}%", mnist_acc_loaded * 100.0);
    }

    Ok(())
}
