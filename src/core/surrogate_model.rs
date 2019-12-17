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

    /// Extend a model with new data, without re-fitting.
    fn extend(
        &self,
        x: Array2<A>,
        y: Array1<A>,
        prior: &Self::Model,
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

    fn predict_mean(&self, x: Array1<A>) -> A {
        let mean = self.predict_mean_a(x.insert_axis(Axis(0)));
        *mean.first().unwrap()
    }

    fn predict_statistics(&self, x: Array1<A>) -> SummaryStatistics<A>;

    fn predict_mean_ei(&self, x: Array1<A>, fmin: A) -> (A, A) {
        let (mean, ei) = self.predict_mean_ei_a(x.insert_axis(Axis(0)), fmin);
        (*mean.first().unwrap(), *ei.first().unwrap())
    }

    /// Predict the mean value of the surrogate model.
    fn predict_mean_a(&self, x: Array2<A>) -> Array1<A>;

    /// Predict the mean value and expected improvement of the surrogate model.
    ///
    /// The expected improvement must be some non-negative number
    /// that must be larger when the opportunity for improvement is larger.
    /// It does not have to be in natural units, but may still be transformed.
    fn predict_mean_ei_a(&self, x: Array2<A>, fmin: A) -> (Array1<A>, Array1<A>);
}

pub struct SummaryStatistics<A> {
    q1: A,
    q2: A,
    q3: A,
    mean: A,
    std: A,
    cv: A,
}

impl<A> SummaryStatistics<A> {
    pub fn new_mean_std_cv_quartiles(
        mean: A,
        std: A,
        cv: A,
        quartiles: [A; 3],
    ) -> SummaryStatistics<A> {
        let [q1, q2, q3] = quartiles;
        SummaryStatistics {
            mean,
            std,
            cv,
            q1,
            q2,
            q3,
        }
    }

    pub fn mean(&self) -> A
    where
        A: Clone,
    {
        self.mean.clone()
    }

    pub fn std(&self) -> A
    where
        A: Clone,
    {
        self.std.clone()
    }

    pub fn cv(&self) -> A
    where
        A: Clone,
    {
        self.cv.clone()
    }

    pub fn q13(&self) -> (A, A)
    where
        A: Clone,
    {
        (self.q1.clone(), self.q3.clone())
    }

    pub fn median(&self) -> A
    where
        A: Clone,
    {
        self.q2.clone()
    }

    pub fn iqr(&self) -> A
    where
        A: std::ops::Sub<Output = A> + Clone,
    {
        self.q3.clone() - self.q1.clone()
    }
}
