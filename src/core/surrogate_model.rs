use ndarray::prelude::*;

use crate::{Scalar, Space, RNG};

/// An estimator that creates a SurrogateModel
pub trait Estimator<A: Scalar> {
    type Model: SurrogateModel<A>;
    type Error: std::fmt::Display + std::fmt::Debug;

    /// Create a default estimator
    fn new(space: &Space) -> Self;

    /// Fit a new model to the given data.
    fn estimate(
        &self,
        x: Array2<A>,
        y: Array1<A>,
        prior: Option<&Self::Model>,
        rng: &mut RNG,
    ) -> Result<Self::Model, Self::Error>;
}

/// A regression model to predict the value of points.
/// This is used to guide the acquisition of new samples.
pub trait SurrogateModel<A: Scalar> {
    /// Length scales for the parameters, estimated by the fitted model.
    ///
    /// Longer length scales indicate less relevant parameters.
    /// By default, the scale is 1.
    /// If no length scale information is available, returns None.
    fn length_scales(&self) -> Vec<f64>;

    fn predict_mean_transformed(&self, x: Array1<A>) -> A {
        let mean = self.predict_mean_transformed_a(x.insert_axis(Axis(0)));
        *mean.first().unwrap()
    }

    fn predict_mean_std_transformed(&self, x: Array1<A>) -> (A, A) {
        let (mean, std) = self.predict_mean_std_transformed_a(x.insert_axis(Axis(0)));
        (*mean.first().unwrap(), *std.first().unwrap())
    }

    fn predict_mean_transformed_a(&self, x: Array2<A>) -> Array1<A>;
    fn predict_mean_std_transformed_a(&self, x: Array2<A>) -> (Array1<A>, Array1<A>);
}
