//! Surrogate Model based on Gaussian Process Regression (GPR)
//!
//! This code is based on GPs as described in Rasmussen & Williams 2006,
//! in particular equations 2.33 and 2.24, and the algorithm 2.1
//! (pseudocode):
//!
//! ``` ignore
//! fn predict(X_train: Matrix,
//!            Y_train: Vector,
//!            kernel: Fn,
//!            noise_level: f64,
//!            X: Matrix) -> ... {
//!     // note on notation: x = A \ b <=> A x = b
//!     let K: Matrix = kernel(x_train, x_train);
//!     let K_trans: Matrix = kernel(x_train, x);
//!
//!     // Line 2:
//!     let L: Matrix = cholesky(k + noise_level * diag);
//!     // Line 3:
//!     let alpha: Vector = L.t() \ (L \ y);
//!     // Line 4:
//!     let y: Vector = K_trans.t() * alpha;
//!     // Line 5:
//!     let for<k> V[..,k] = L \ K_trans[..,k]
//!     // Line 6:
//!     let for<k> y_var[k] = kernel(X, X)[k,k] - sum(for<i> V[i,k] * V[i,k]);
//!     // Line 7:
//!     let lml = log(p(y_train | x_train))
//!             = - 1/2 * y_train.t() * alpha - sum(for<i> log(L[i,i])) - n/2 * log(2*pi);
//!     // Line 8:
//!     return (y, y_var, lml);
//! }
//! ```
//!
//! Large parts of this code are based on skopt.learning.GaussianProcessRegressor
//! from skopt (scikit-optimize)
//! at https://github.com/scikit-optimize/scikit-optimize
//!
//! See also the Python variant ggga
//! at https://github.com/latk/ggga.py

use ndarray::prelude::*;
use ndarray_stats as ndstats;
use noisy_float;
use num_traits::Float;

#[cfg(test)]
use speculate::speculate;

use crate::gpr::*;
use crate::util::{BoundedValue, BoundsError};
use crate::{Estimator, Space, SurrogateModel, RNG};

type ConcreteKernel = Product<ConstantKernel, Matern>;

#[derive(Debug, Clone)]
pub struct SurrogateModelGPR<A: Scalar> {
    kernel: ConcreteKernel,
    noise: BoundedValue<f64>,
    x_train: Array2<A>,
    y_train: Array1<A>,
    alpha: Array1<A>,
    k_inv: Array2<A>,
    y_norm: YNormalization<A>,
    lml: f64,
    space: Space,
}

impl<A: Scalar> SurrogateModelGPR<A> {
    pub fn kernel(&self) -> &ConcreteKernel {
        &self.kernel
    }
}

impl<A: Scalar> SurrogateModel<A> for SurrogateModelGPR<A> {
    fn space(&self) -> &Space {
        &self.space
    }

    fn length_scales(&self) -> Vec<f64> {
        self.kernel
            .k2()
            .length_scale()
            .iter()
            .map(BoundedValue::value)
            .collect()
    }

    fn predict_mean_transformed_a(&self, x: Array2<A>) -> Array1<A> {
        let y = predict(
            &self.kernel,
            self.alpha.view(),
            x.view(),
            self.x_train.view(),
            self.k_inv.view(),
            None,
        );

        self.y_norm.y_from_normalized(y)
    }

    fn predict_mean_std_transformed_a(&self, x: Array2<A>) -> (Array1<A>, Array1<A>) {
        let mut y_var = Array1::zeros(x.rows());
        let y = predict(
            &self.kernel,
            self.alpha.view(),
            x.view(),
            self.x_train.view(),
            self.k_inv.view(),
            Some(y_var.view_mut()),
        );

        (
            self.y_norm.y_from_normalized(y),
            self.y_norm.y_std_from_normalized_variance(y_var),
        )
    }
}

impl<A: Scalar> Estimator<A> for EstimatorGPR {
    type Model = SurrogateModelGPR<A>;
    type Error = Error;

    fn new(space: &Space) -> Self {
        let noise_bounds = (1e-5, 1e5);
        let length_scale_bounds = std::iter::repeat((1e-3, 1e3)).take(space.len()).collect();
        let n_restarts_optimizer = 2;
        let matern_nu = 5. / 2.;
        let amplitude_bounds = None;
        EstimatorGPR {
            noise_bounds,
            length_scale_bounds,
            n_restarts_optimizer,
            matern_nu,
            amplitude_bounds,
        }
    }

    fn estimate(
        &self,
        x: Array2<A>,
        y: Array1<A>,
        space: Space,
        prior: Option<&Self::Model>,
        rng: &mut RNG,
    ) -> Result<Self::Model, Self::Error> {
        let (n_observations, n_features) = x.dim();
        assert!(
            y.len() == n_observations,
            "expected y values for {} observations: {}",
            n_observations,
            y,
        );
        assert!(
            space.len() == n_features,
            "number of parameters ({}) must equal number of features ({})",
            space.len(),
            n_features,
        );

        let n_restarts_optimizer = self.n_restarts_optimizer;

        let (y_train, y_norm) = YNormalization::normalized_from_y(y);
        let x_train = space.transform_sample_a(x);

        let amplitude = estimate_amplitude(y_train.view(), self.amplitude_bounds);

        let (kernel, noise) = get_kernel_or_default(prior, amplitude, self)?;

        let FittedKernel {
            kernel,
            noise,
            alpha,
            k_inv,
            lml,
        } = FittedKernel::new(
            kernel,
            x_train.clone(),
            y_train.clone(),
            &mut rng.fork_random_state(),
            n_restarts_optimizer,
            noise,
        );

        Ok(SurrogateModelGPR {
            kernel,
            noise,
            x_train,
            y_train,
            alpha,
            k_inv,
            y_norm,
            lml,
            space,
        })
    }
}

#[derive(Debug)]
pub struct EstimatorGPR {
    noise_bounds: (f64, f64),
    length_scale_bounds: Vec<(f64, f64)>,
    n_restarts_optimizer: usize,
    matern_nu: f64,
    amplitude_bounds: Option<(f64, f64)>,
}

impl EstimatorGPR {
    pub fn noise_bounds(self, lo: f64, hi: f64) -> Self {
        EstimatorGPR {
            noise_bounds: (lo, hi),
            ..self
        }
    }

    pub fn length_scale_bounds(self, bounds: Vec<(f64, f64)>) -> Self {
        EstimatorGPR {
            length_scale_bounds: bounds,
            ..self
        }
    }

    pub fn n_restarts_optimizer(self, n: usize) -> Self {
        EstimatorGPR {
            n_restarts_optimizer: n,
            ..self
        }
    }

    pub fn matern_nu(self, nu: f64) -> Self {
        EstimatorGPR {
            matern_nu: nu,
            ..self
        }
    }

    pub fn amplitude_bounds(self, bounds: Option<(f64, f64)>) -> Self {
        EstimatorGPR {
            amplitude_bounds: bounds,
            ..self
        }
    }
}

fn get_kernel_or_default<A: Scalar>(
    prior: Option<&SurrogateModelGPR<A>>,
    amplitude: BoundedValue<f64>,
    config: &EstimatorGPR,
) -> Result<(ConcreteKernel, BoundedValue<f64>), Error> {
    if let Some(prior) = prior {
        return Ok((prior.kernel.clone(), prior.noise.clone()));
    }

    let noise = BoundedValue::new(1.0, config.noise_bounds.0, config.noise_bounds.1)
        .map_err(Error::NoiseBounds)?;

    let length_scale = config
        .length_scale_bounds
        .iter()
        .map(|&(lo, hi)| BoundedValue::new(((lo.ln() + hi.ln()) / 2.).exp(), lo, hi))
        .collect::<Result<_, _>>()
        .map_err(Error::LengthScaleBounds)?;

    let amplitude = ConstantKernel::new(amplitude);
    // TODO: adjust length scale bounds
    let main_kernel = Matern::new(config.matern_nu, length_scale);
    let kernel = Product::of(amplitude, main_kernel);

    Ok((kernel, noise))
}

#[derive(Debug, PartialEq, Clone)]
struct YNormalization<A: Scalar> {
    amplitude: A,
    expected: A,
}

impl<A: Scalar> YNormalization<A> {
    fn fudge_min() -> A {
        A::from_f(0.05)
    }

    fn normalized_from_y(mut y: Array1<A>) -> (Array1<A>, Self) {
        use ndarray_stats::QuantileExt;

        // shift the values so that all are positive
        let expected = *y.min().unwrap();
        y -= expected;

        // scale the values so that the mean is set to 1
        let amplitude = y.mean_axis(Axis(0)).into_scalar();
        let amplitude = if amplitude <= A::from_f(0.0) {
            A::from_f(1.0)
        } else {
            amplitude
        };
        y /= amplitude;

        // raise the minimum slightly above zero
        y += Self::fudge_min();

        (
            y,
            Self {
                amplitude,
                expected,
            },
        )
    }

    fn y_from_normalized(&self, mut y: Array1<A>) -> Array1<A> {
        y -= Self::fudge_min();
        y *= self.amplitude;
        y += self.expected;
        y
    }

    fn y_std_from_normalized_variance(&self, y_variance: Array1<A>) -> Array1<A> {
        y_variance.mapv_into(Float::sqrt) * self.amplitude
    }
}

#[cfg(test)]
speculate! {
    describe "struct YNormalization" {

        it "has an inverse operation" {
            let original: Array1<f64> = vec![1., 2., 3., 4.].into();
            let (y, norm) = YNormalization::normalized_from_y(original.clone());
            let result = norm.y_from_normalized(y);
            assert!(original.all_close(&result, 0.0001));
        }

        it "also works with negative numbers" {
            let original: Array1<f64> = vec![-5., 3., 8., -2.].into();
            let (y, norm) = YNormalization::normalized_from_y(original.clone());
            let result = norm.y_from_normalized(y);
            assert!(original.all_close(&result, 0.0001));
        }
    }
}

fn estimate_amplitude<A: Scalar>(
    y: ArrayView1<A>,
    bounds: Option<(f64, f64)>,
) -> BoundedValue<f64> {
    use ndstats::Quantile1dExt;
    use noisy_float::types::N64;
    let (lo, hi) = bounds.unwrap_or_else(|| {
        let hi = y.mapv(|x| x.powi(2)).sum().into();
        let lo = y
            .mapv(|x| N64::from_f64(x.into()))
            .quantile_mut(N64::from_f64(0.1), &ndstats::interpolate::Lower)
            .unwrap()
            .raw()
            .powi(2)
            * y.len() as f64;
        assert!(lo >= 0.0);
        let lo = if lo > 2e-5 { lo } else { 2e-5 };
        (lo / 2.0, hi * 2.0)
    });
    let start = Array1::from_vec(vec![lo, hi])
        .mapv_into(f64::ln)
        .mean_axis(Axis(0))
        .into_scalar()
        .exp();
    BoundedValue::new(start, lo, hi).unwrap()
}

#[derive(Debug)]
pub enum Error {
    NoiseBounds(BoundsError<f64>),
    LengthScaleBounds(BoundsError<f64>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Error::NoiseBounds(BoundsError { value, min, max }) => write!(
                f,
                "noise level {} violated bounds [{}, {}] during model fitting",
                value, min, max,
            ),
            Error::LengthScaleBounds(BoundsError { value, min, max }) => write!(
                f,
                "length scale {} violated bounds [{}, {}] during model fitting",
                value, min, max,
            ),
        }
    }
}