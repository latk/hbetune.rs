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

use crate::core::ynormalize::{Projection, YNormalize};
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
    y_norm: YNormalize<A>,
    lml: f64,
}

impl<A: Scalar> SurrogateModelGPR<A> {
    pub fn kernel(&self) -> &ConcreteKernel {
        &self.kernel
    }
}

impl<A: Scalar> SurrogateModel<A> for SurrogateModelGPR<A> {
    fn length_scales(&self) -> Vec<f64> {
        self.kernel
            .k2()
            .length_scale()
            .iter()
            .map(BoundedValue::value)
            .collect()
    }

    fn predict_mean_a(&self, x: Array2<A>) -> Array1<A> {
        let y = predict(
            &self.kernel,
            self.alpha.view(),
            x.view(),
            self.x_train.view(),
            self.k_inv.view(),
            None,
        );

        self.y_norm.project_location_from_normalized(y)
    }

    fn predict_statistics(
        &self,
        x: Array1<A>,
    ) -> crate::core::surrogate_model::SummaryStatistics<A> {
        use crate::core::surrogate_model::SummaryStatistics;
        use statrs::distribution::InverseCDF as _;

        let mut vnorm = Array1::zeros(1);
        let mnorm = predict(
            &self.kernel,
            self.alpha.view(),
            x.view().insert_axis(Axis(0)),
            self.x_train.view(),
            self.k_inv.view(),
            Some(vnorm.view_mut()),
        );

        let vnorm_scalar = vnorm
            .first()
            .expect("should contain one element")
            .sqrt()
            .into();
        let mnorm_scalar = (*mnorm.first().expect("should contain one element")).into();
        let distnorm = if approx_eq!(f64, vnorm_scalar, 0.0) {
            None
        } else {
            match statrs::distribution::Normal::new(mnorm_scalar, vnorm_scalar) {
                Ok(distribution) => Some(distribution),
                Err(err) => panic!(
                    "could not create normal distribution with mean {} std {}: {}",
                    mnorm_scalar, vnorm_scalar, err
                ),
            }
        };

        let mean = *self
            .y_norm
            .project_mean_from_normalized(mnorm.clone(), vnorm.view())
            .first()
            .unwrap();
        let std = *self
            .y_norm
            .project_std_from_normalized(mnorm.view(), vnorm.clone())
            .first()
            .unwrap();
        let cv = *self
            .y_norm
            .project_cv_from_normalized(mnorm.view(), vnorm)
            .first()
            .unwrap();

        let q123norm = if let Some(distnorm) = distnorm {
            array![0.25, 0.5, 0.75].mapv(|x| A::from_f(distnorm.inverse_cdf(x)))
        } else {
            array![mnorm_scalar, mnorm_scalar, mnorm_scalar].mapv(A::from_f)
        };
        let q123 = self.y_norm.project_location_from_normalized(q123norm);
        let quartiles = match q123.to_vec().as_slice() {
            [q1, q2, q3] => [*q1, *q2, *q3],
            _ => unreachable!(),
        };

        SummaryStatistics::new_mean_std_cv_quartiles(mean, std, cv, quartiles)
    }

    fn predict_mean_ei_a(&self, x: Array2<A>, fmin: A) -> (Array1<A>, Array1<A>) {
        use crate::core::acquisition::expected_improvement;

        let mut y_var = Array1::zeros(x.nrows());
        let y = predict(
            &self.kernel,
            self.alpha.view(),
            x.view(),
            self.x_train.view(),
            self.k_inv.view(),
            Some(y_var.view_mut()),
        );

        let fmin = *self
            .y_norm
            .project_into_normalized(array![fmin])
            .first()
            .unwrap();

        let mut ei: Array1<A> = Array1::zeros(x.nrows());
        ndarray::Zip::from(&mut ei)
            .and(&y)
            .and(&y_var)
            .apply(|ei, &y, &var| {
                *ei = A::from_f(expected_improvement(
                    y.into(),
                    var.sqrt().into(),
                    fmin.into(),
                ))
            });

        let y = self.y_norm.project_location_from_normalized(y);
        (y, ei)
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
        let y_projection = Projection::Linear;
        EstimatorGPR {
            noise_bounds,
            length_scale_bounds,
            n_restarts_optimizer,
            matern_nu,
            amplitude_bounds,
            y_projection,
        }
    }

    fn estimate(
        &self,
        x: Array2<A>,
        y: Array1<A>,
        prior: Option<&Self::Model>,
        rng: &mut RNG,
    ) -> Result<Self::Model, Self::Error> {
        let (n_observations, _n_features) = x.dim();
        assert!(
            y.len() == n_observations,
            "expected y values for {} observations: {}",
            n_observations,
            y,
        );

        let n_restarts_optimizer = self.n_restarts_optimizer;

        let (y_train, y_norm) = YNormalize::new_project_into_normalized(y, self.y_projection);
        let x_train = x;

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
    y_projection: Projection,
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

    pub fn y_projection(self, y_projection: Projection) -> Self {
        Self {
            y_projection,
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

fn estimate_amplitude<A: Scalar>(
    y: ArrayView1<A>,
    bounds: Option<(f64, f64)>,
) -> BoundedValue<f64> {
    use ndstats::Quantile1dExt as _;
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
    let start = f64::exp((lo.ln() + hi.ln()) / 2.0);
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
