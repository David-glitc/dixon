//! Metrics for evaluating neural network performance.
/// Accuracy
pub fn accuracy(dataset: &[(Vec<f64>, Vec<f64>)], model: &crate::network::MLP) -> f64 {
    let mut correct = 0;
    for (input, target) in dataset {
        let pred = model.predict(input);
        let pred_class = pred.iter().enumerate().fold(0usize, |max_i, (i, &v)| if v > pred[max_i] { i } else { max_i });
        let true_class = target.iter().position(|&x| (x - 1.0).abs() < 1e-6).unwrap_or(0);
        if pred_class == true_class {
            correct += 1;
        }
    }
    correct as f64 / dataset.len() as f64
}

/// Simple confusion matrix (for small num_classes)
pub fn confusion_matrix(dataset: &[(Vec<f64>, Vec<f64>)], model: &crate::network::MLP, num_classes: usize) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0; num_classes]; num_classes];
    for (input, target) in dataset {
        let pred = model.predict(input);
        let pred_class = pred.iter().enumerate().fold(0usize, |max_i, (i, &v)| if v > pred[max_i] { i } else { max_i });
        let true_class = target.iter().position(|&x| (x - 1.0).abs() < 1e-6).unwrap_or(0);
        cm[true_class][pred_class] += 1;
    }
    cm
}