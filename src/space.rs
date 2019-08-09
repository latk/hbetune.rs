use ndarray::prelude::*;

use crate::{Individual, Scalar, RNG};

/// The parameter space that contains feasible solutions.
/// This space is usually handled in its natural units,
/// but internally each parameter can be projected into the range 0 to 1 inclusive.
#[derive(Debug, Clone, Default)]
pub struct Space {
    params: Vec<Parameter>,
}

impl Space {
    /// Create a new parameter space.
    pub fn new() -> Self {
        Space::default()
    }

    /// The number of dimensions.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    // The space is empty if no parameters were provided.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// The parameters, in order.
    pub fn params(&self) -> &[Parameter] {
        self.params.as_slice()
    }

    /// Add a real-valued parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// Panics if the parameter range has zero size.
    pub fn add_real_parameter(&mut self, name: impl Into<String>, lo: f64, hi: f64) {
        assert!(lo < hi);
        self.add_parameter(Parameter::Real {
            name: name.into(),
            lo,
            hi,
        });
    }

    pub fn add_parameter(&mut self, param: Parameter) {
        self.params.push(param);
    }

    /// Project a parameter array in-place into the range 0 to 1 inclusive.
    pub fn transform_sample_inplace<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            param.transform_sample_inplace(x);
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
    /// assert_eq!(space.transform_sample(array![-1.0, 7.0]),
    ///            array![0.25, 0.7]);
    /// # }
    /// ```
    pub fn transform_sample<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.transform_sample_inplace(x.view_mut());
        x
    }

    /// Project a matrix of parameter vectors into the range 0 to 1 inclusive.
    pub fn transform_sample_a<A: Scalar>(&self, mut xs: Array2<A>) -> Array2<A> {
        for x in xs.outer_iter_mut() {
            self.transform_sample_inplace(x);
        }
        xs
    }

    /// Get a parameter vector from a projection in the range 0 to 1 inclusive.
    pub fn untransform_sample_inplace<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            param.untransform_sample_inplace(x)
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
    /// assert_eq!(space.untransform_sample(space.transform_sample(params.clone())),
    ///            params);
    /// # }
    /// ```
    pub fn untransform_sample<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.untransform_sample_inplace(x.view_mut());
        x
    }

    /// Sample a new Individual uniformly from the projected parameter space.
    pub fn sample<A>(&self, rng: &mut RNG) -> Individual<A>
    where
        A: Scalar + rand::distributions::uniform::SampleUniform,
    {
        let range = num_traits::zero()..=num_traits::one();
        let sample = Array::from_shape_fn(self.len(), |_| rng.uniform(range.clone()));
        Individual::new(self.untransform_sample(sample))
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
                Parameter::Real { .. } => {
                    A::from_f(sample_truncnorm((*x).into(), relscale[i], 0.0, 1.0, rng))
                }
            }
        }

        sample_transformed
    }
}

fn sample_truncnorm(mu: f64, sigma: f64, a: f64, b: f64, rng: &mut RNG) -> f64 {
    use statrs::distribution::{InverseCDF, Normal, Univariate};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let az = (a - mu) / sigma;
    let bz = (b - mu) / sigma;

    let xz = rng.uniform(normal.cdf(az)..=normal.cdf(bz));

    normal.inverse_cdf(xz) * sigma + mu
}

#[derive(Debug, Clone)]
pub enum Parameter {
    Real { name: String, lo: f64, hi: f64 },
}

impl Parameter {
    pub fn name(&self) -> &str {
        match self {
            Parameter::Real { name, .. } => name,
        }
    }

    fn transform_sample_inplace<A: Scalar>(&self, x: &mut A) {
        *x = match *self {
            Parameter::Real { lo, hi, .. } => (*x - A::from_f(lo)) / A::from_f(hi - lo),
        };
    }

    fn untransform_sample_inplace<A: Scalar>(&self, x: &mut A) {
        *x = match *self {
            Parameter::Real { lo, hi, .. } => *x * A::from_f(hi - lo) + A::from_f(lo),
        };
    }
}

impl std::str::FromStr for Parameter {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Parameter, Self::Err> {
        use failure::ResultExt as _;
        let err_too_few_items =
            || format_err!("too few items, expected: '<name> <type> <...>' but got: {}", s);
        let err_real_too_few_items =
            || format_err!("too few items, expected: '<name> real <lo> <hi>' but got: {}", s);
        let err_real_too_many_items =
            || format_err!("too many items, expected: '<name> real <lo> <hi>' but got: {}", s);

        let mut items = s.split_whitespace();
        let name: String = items
            .next()
            .ok_or_else(err_too_few_items)?
            .to_owned();
        let the_type: &str = items
            .next()
            .ok_or_else(err_too_few_items)?;
        match the_type {
            "real" => {
                let lo = items
                    .next()
                    .ok_or_else(err_real_too_few_items)?
                    .parse::<f64>()
                    .with_context(|err| format!("while parsing <lo>: {}", err))?;

                let hi = items
                    .next()
                    .ok_or_else(err_real_too_few_items)?
                    .parse::<f64>()
                    .with_context(|err| format!("while parsing <hi>: {}", err))?;

                if !(items.next().is_none()) {
                    return Err(err_real_too_many_items());
                }

                Ok(Parameter::Real { name, lo, hi })
            }
            t => bail!("type must be 'real', was: {}", t),
        }
    }
}

