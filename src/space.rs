use ndarray::prelude::*;

use crate::kernel::Scalar;
use crate::individual::Individual;
use crate::random::RNG;

/// The parameter space that contains feasible solutions.
/// This space is usually handled in its natural units,
/// but internally each parameter can be projected into the range 0 to 1 inclusive.
#[derive(Debug, Clone)]
pub struct Space {
    params: Vec<Parameter>,
}

impl Space {
    /// Create a new parameter space.
    pub fn new() -> Self {
        let params = Vec::new();
        Space { params }
    }

    /// The number of dimensions.
    pub fn len(&self) -> usize { self.params.len() }

    /// Add a real-valued parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// Panics if the parameter range has zero size.
    pub fn add_real_parameter(&mut self, name: impl Into<String>, lo: f64, hi: f64) {
        assert!(lo < hi);
        self.params.push(Parameter::Real { name: name.into(), lo, hi });
    }

    /// Project a parameter array in-place into the range 0 to 1 inclusive.
    pub fn into_transformed_mut<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            param.into_transformed_mut(x);
        }
    }

    /// Project a parameter array into the range 0 to 1 inclusive.
    ///
    /// ```
    /// #[macro_use] extern crate ndarray;
    /// # use ggtune::Space;
    /// # fn main() {
    /// let mut space = Space::new();
    /// space.add_real_parameter("a", -2.0, 2.0);
    /// space.add_real_parameter("b", 0.0, 10.0);
    /// assert_eq!(space.into_transformed(array![-1.0, 7.0]),
    ///            array![0.25, 0.7]);
    /// # }
    /// ```
    pub fn into_transformed<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.into_transformed_mut(x.view_mut());
        x
    }

    /// Project a matrix of parameter vectors into the range 0 to 1 inclusive.
    pub fn into_transformed_a<A: Scalar>(&self, mut xs: Array2<A>) -> Array2<A> {
        for x in xs.outer_iter_mut() {
            self.into_transformed_mut(x);
        }
        xs
    }

    /// Get a parameter vector from a projection in the range 0 to 1 inclusive.
    pub fn from_transformed_mut<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            param.from_transformed_mut(x)
        }
    }

    /// Get a parameter vector from a projection in the range 0 to 1 inclusive.
    ///
    /// ```
    /// #[macro_use] extern crate ndarray;
    /// # use ggtune::Space;
    /// # fn main() {
    /// let mut space = Space::new();
    /// space.add_real_parameter("a", -2.0, 2.0);
    /// space.add_real_parameter("b", 0.0, 10.0);
    /// let params = array![-1.0, 7.0];
    /// assert_eq!(space.from_transformed(space.into_transformed(params.clone())),
    ///            params);
    /// # }
    /// ```
    pub fn from_transformed<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.from_transformed_mut(x.view_mut());
        x
    }

    /// Sample a new Individual uniformly from the projected parameter space.
    pub fn sample<A>(&self, rng: &mut RNG) -> Individual<A>
    where A: Scalar + rand::distributions::uniform::SampleUniform
    {
        let range = num_traits::zero() ..= num_traits::one();
        let sample = Array::from_shape_fn(self.len(), |_| rng.uniform(range.clone()));
        Individual::new(self.from_transformed(sample))
    }

    pub fn mutate_transformed<A: Scalar>(
        &self,
        mut sample_transformed: Array1<A>,
        relscale: &[f64],
        rng: &mut RNG,
    ) -> Array1<A> {
        assert_eq!(sample_transformed.len(), relscale.len());
        assert_eq!(sample_transformed.len(), self.len());

        for (i, x) in sample_transformed.indexed_iter_mut() {
            *x = match self.params[i] {
                Parameter::Real { .. } =>
                    A::from_f(sample_truncnorm((*x).into(), relscale[i], 0.0, 1.0, rng)),
            }
        }

        sample_transformed
    }
}

fn sample_truncnorm(mu: f64, sigma: f64, a: f64, b: f64, rng: &mut RNG) -> f64 {
    use statrs::distribution::{Normal, Univariate, InverseCDF};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let az = (a - mu)/sigma;
    let bz = (b - mu)/sigma;

    let xz = rng.uniform(normal.cdf(az) ..= normal.cdf(bz));

    let x = normal.inverse_cdf(xz) * sigma + mu;

    x
}

#[derive(Debug, Clone)]
enum Parameter {
    Real { name: String, lo: f64, hi: f64 },
}

impl Parameter {
    fn into_transformed_mut<A: Scalar>(&self, x: &mut A) {
        *x = match *self {
            Parameter::Real { name: _, lo, hi } => (*x - A::from_f(lo)) / A::from_f(hi - lo),
        };
    }

    fn from_transformed_mut<A: Scalar>(&self, x: &mut A) {
        *x = match *self {
            Parameter::Real { name: _, lo, hi } => *x * A::from_f(hi - lo) + A::from_f(lo),
        };
    }
}
