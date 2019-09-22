use ndarray::prelude::*;

use crate::gpr::Scalar;
use crate::util::BoundsError;

/// A kernel function that can be used in a Gaussian Process
/// and can be tuned.
pub trait Kernel: Clone + std::fmt::Debug {
    /// Evaluate the kernel function.
    /// Input arrays have shape (n_samples, n_features).
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A>;

    /// Get the gradients of the kernel function with respect to the hyperparameters theta.
    /// Input array has shape (n_samples, n_features).
    fn theta_grad<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>);

    /// Get the diagonal of the kernel function in a certain position.
    /// Input array has shape (n_samples, n_features).
    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A>;

    /// Number of tunable parameters.
    fn n_params(&self) -> usize;

    /// Get the log-transformed parameters, useful for tuning.
    fn theta(&self) -> Vec<f64>;

    // Update the kernel parameters with a theta vector.
    // Returns None if the theta is unacceptable, e.g. because it violates bounds.
    // Panics if the theta has wrong length – must equal n_params().
    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>>
    where
        Self: Sized;

    /// Update the kernel parameters with a theta vector.
    /// Adjust the parameters if they would otherwise violate bounds.
    /// Panics if the theta vector has wrong lenght – must equal n_params().
    fn with_clamped_theta(self, theta: &[f64]) -> Self
    where
        Self: Sized;

    // Get the theta bounds for tuning (log-transformed).
    fn bounds(&self) -> Vec<(f64, f64)>;
}
