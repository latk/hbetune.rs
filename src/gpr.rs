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

use crate::kernel::{BoundedValue, BoundsError, ConstantKernel, Kernel, Matern, Product};
use crate::util::DisplayIter;
use crate::{Estimator, Scalar, Space, SurrogateModel, RNG};

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
        let y = basic_predict(
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
        let y = basic_predict(
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

        let (mut kernel, noise) = get_kernel_or_default(prior, amplitude, self)?;

        let KernelFittingResult {
            noise,
            alpha,
            k_inv,
            lml,
        } = fit_kernel(
            &mut kernel,
            x_train.clone(),
            y_train.clone(),
            &mut rng.fork_random_state(),
            n_restarts_optimizer,
            noise,
        )?;

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

fn basic_predict<A>(
    kernel: &impl Kernel,
    alpha: ArrayView1<A>,
    x: ArrayView2<A>,
    x_train: ArrayView2<A>,
    k_inv: ArrayView2<A>,
    want_variance: Option<ArrayViewMut1<A>>,
) -> Array1<A>
where
    A: Scalar,
{
    let k_trans: Array2<A> = kernel.kernel(x.view(), x_train);
    let y = k_trans.dot(&alpha);

    if let Some(mut variance_out) = want_variance {
        // Compute variance of predictive distribution.
        // The "min_noise" term is added to avoid negative varainces
        // (could occur due to numeric issues).
        let min_noise = A::from_f(1e-5);
        // y_var = diag(kernel(X)) + min_noise - einsum("ki,kj,ij->k", k_trans, k_trans, k_inv)
        // y_var[k] = diag(kernel(X))[k] + min_noise - sum(outer(k_trans[k], k_trans[k]) * k_inv)
        // hmm, but sklearn has this:
        // y_var[k] = diag(kernel(X))[k] - sum(for<i> dot(k_trans, k_inv)[i] * k_trans[i])
        let mut y_var = kernel.diag(x.view()) + min_noise
            - Array::from_iter(
                k_trans
                    .dot(&k_inv)
                    .outer_iter()
                    .zip(k_trans.outer_iter())
                    .map(|(a, b)| a.dot(&b)),
            );

        if let Some(elements_below_threshold) =
            clamp_negative_variance(y_var.view_mut(), -min_noise.sqrt())
        {
            eprintln!(
                "Variances below 0 were predicted and will be corrected: {:.2e}",
                DisplayIter::from(elements_below_threshold),
            );
        }

        variance_out.assign(&y_var);
    }

    y
}

#[cfg(test)]
speculate! {
    describe "fn basic_predict" {
        it "predicts a simple case" {
            // problem definition
            let xs = array![[0.0],[0.5],[0.5],[1.0]];
            let ys = array![0.0, 0.8, 1.2, 2.0];
            let mut kernel = Product::of(
                ConstantKernel::new(BoundedValue::new(3.0, 0.1, 4.0).unwrap()),
                Matern::new(2.5, vec![BoundedValue::new(1.5, 0.1, 2.0).unwrap()]));
            const SEED: usize = 938274;
            const N_RESTARTS_OPTIMIZER: usize = 4;
            let mut rng = crate::random::RNG::new_with_seed(SEED);

            // precompute stuff
            let KernelFittingResult { alpha, k_inv, .. } = fit_kernel(
                &mut kernel, xs.clone(), ys.clone(), &mut rng, N_RESTARTS_OPTIMIZER,
                BoundedValue::new(1.0, 0.001, 1.0).unwrap(),
            ).expect("fit_kernel() should succeed");

            // perform prediction
            let predict_xs = array![[0.0],[0.25],[0.5],[0.75],[1.0]];
            let mut variances = Array1::zeros(5);
            let prediction = basic_predict(
                &kernel, alpha.view(), predict_xs.view(), xs.view(), k_inv.view(),
                Some(variances.view_mut()));

            assert_all_close!(prediction, array![0.0, 0.5, 1.0, 1.5, 2.0], 0.1);
            assert_all_close!(variances, Array1::from_elem(5, 0.03), 0.03);
        }
    }
}

/// Check if any of the variances is negative because of numerical issues.
/// If yes: set the variance to 0.
/// But only warn on largeish differences.
fn clamp_negative_variance<A: Scalar>(
    mut variances: ArrayViewMut1<A>,
    warning_level: A,
) -> Option<Vec<A>> {
    let elements_below_threshold: Vec<_> = variances
        .iter()
        .cloned()
        .filter(|x| *x < warning_level)
        .collect();

    let zero = A::zero();

    variances.map_inplace(|x| {
        if *x < zero {
            *x = zero;
        }
    });

    if elements_below_threshold.is_empty() {
        None
    } else {
        Some(elements_below_threshold)
    }
}

#[cfg(test)]
speculate! {
    describe "fn clamp_negative_variance()" {
        it "returns elements with very negative variances" {
            let mut variances = array![1., -2., -0.5];
            assert_eq!(clamp_negative_variance(variances.view_mut(), -1.), Some(vec![-2.]));
            assert_eq!(variances, array![1., 0., 0.]);
        }

        it "does not return element with mild negative variances" {
            let mut variances = array![1., 2., -0.5];
            assert_eq!(clamp_negative_variance(variances.view_mut(), -1.), None);
            assert_eq!(variances, array![1., 2., 0.]);
        }
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

struct KernelFittingResult<A> {
    noise: BoundedValue<f64>,
    alpha: Array1<A>,
    k_inv: Array2<A>,
    lml: f64,
}

/// Assign kernel parameters with maximal log-marginal likelihood.
///
/// The current kernel parameters (theta) are used as an optimization starting point.
fn fit_kernel<A: Scalar>(
    kernel: &mut impl Kernel,
    x_train: Array2<A>,
    y_train: Array1<A>,
    rng: &mut RNG,
    n_restarts_optimizer: usize,
    noise: BoundedValue<f64>,
) -> Result<KernelFittingResult<A>, Error> {
    let n_observations = x_train.rows();
    assert!(y_train.dim() == n_observations);

    struct Capture<A: Scalar> {
        lml: f64,
        theta: Vec<f64>,
        noise: f64,
        mat_k_factorization: CholeskyFactorizedArray<A>,
        alpha: Array1<A>,
    };

    // calculate the negative log-marginal likelihood of a theta-vector
    // and its gradients
    let capture: std::cell::RefCell<Option<Capture<A>>> = None.into();
    let obj_func = |theta: &[f64], want_gradient: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let (noise_theta, kernel_theta) = theta.split_first().unwrap();
        let kernel = kernel.clone().with_clamped_theta(kernel_theta);
        let noise: A = A::from_f(noise_theta.exp());

        let LmlResult {
            lml,
            lml_gradient,
            alpha,
            factorization,
        } = match lml_with_gradients(kernel, noise, x_train.view(), y_train.view()) {
            Some(result) => result,
            None => {
                if let Some(grad) = want_gradient {
                    for g in grad {
                        *g = 0.0;
                    }
                }
                return std::f64::INFINITY;
            }
        };

        // capture optimal theta incl already computed matrices
        let caplml = capture.borrow().as_ref().map(|cap| cap.lml);
        if caplml.map(|caplml| lml > caplml).unwrap_or(true) {
            capture.replace(Some(Capture {
                lml,
                theta: kernel_theta.to_vec(),
                noise: noise.into(),
                mat_k_factorization: factorization,
                alpha,
            }));
        }

        // negate the lml for minimization and return
        if let Some(gradient) = want_gradient {
            for (out, lg) in gradient.iter_mut().zip(lml_gradient) {
                *out = -lg;
            }
        }
        -lml
    };

    // Perform multiple optimization runs.
    // Usually these functions return the output,
    // but here we capture the results in the objective function.

    let mut bounds = vec![(noise.min().ln(), noise.max().ln())];
    bounds.extend(kernel.bounds());

    let mut initial_theta = vec![noise.value().ln()];
    initial_theta.extend(kernel.theta());

    crate::util::minimize_by_gradient_with_restarts(
        &obj_func,
        initial_theta.as_mut_slice(),
        bounds.as_slice(),
        (),
        n_restarts_optimizer,
        rng,
    );

    let Capture {
        theta,
        noise: captured_noise,
        mat_k_factorization,
        alpha,
        lml,
    } = capture.replace(None).unwrap();

    *kernel = kernel.clone().with_clamped_theta(theta.as_slice());
    let noise = noise.with_clamped_value(captured_noise);

    // Precompute arrays needed at prediction
    use ndarray_linalg::cholesky::*;
    let mat_k_inv = mat_k_factorization.invc_into().unwrap();
    Ok(KernelFittingResult {
        noise,
        alpha,
        k_inv: mat_k_inv,
        lml,
    })
}

type CholeskyFactorizedArray<A> =
    ndarray_linalg::cholesky::CholeskyFactorized<ndarray::OwnedRepr<A>>;

struct LmlResult<A> {
    lml: f64,
    lml_gradient: Vec<f64>,
    alpha: Array1<A>,
    factorization: CholeskyFactorizedArray<A>,
}

/// calculate the log marginal likelihood and gradients
/// of a certain kernel/noise configuration
fn lml_with_gradients<A: Scalar>(
    kernel: impl Kernel,
    noise: A,
    x_train: ArrayView2<A>,
    y_train: ArrayView1<A>,
) -> Option<LmlResult<A>> {
    use ndarray_linalg::cholesky::*;

    // calculate combined gradient
    let (mut kernel_matrix, kernel_gradient) = kernel.gradient(x_train);
    let noise_gradient = Array2::eye(x_train.rows()).insert_axis(Axis(2)) * noise;
    let gradient = stack!(Axis(2), noise_gradient, kernel_gradient);

    // add noise to kernel matrix
    kernel_matrix.diag_mut().map_mut(|x| *x += noise);

    // find the Cholesky decomposition
    let factorization = match kernel_matrix.factorizec(UPLO::Lower) {
        Ok(lower) => lower,
        Err(_) => return None,
    };

    // solve the system "K alpha = y" for alpha
    // based on the Cholesky factorization
    let alpha = factorization.solvec(&y_train).unwrap();
    assert!(alpha.dim() == y_train.len());

    let lml = -0.5 * y_train.dot(&alpha).into()
        - factorization.factor.diag().mapv(Float::ln).sum().into()
        - y_train.len() as f64 / 2.0 * (2.0 * std::f64::consts::PI).ln();

    let lml_gradient = {
        let tmp = outer(alpha.view(), alpha.view()) - &factorization.invc().unwrap();
        // compute "0.5 * trace(tmp dot kernel_gradient)"
        // without constructing the full matrix
        // as only the diagonal is required
        gradient
            .axis_iter(Axis(2))
            .map(|grad| 0.5 * (&tmp * &grad).sum().into())
            .collect()
    };

    Some(LmlResult {
        lml,
        lml_gradient,
        alpha,
        factorization,
    })
}

fn outer<A: Scalar>(a: ArrayView1<A>, b: ArrayView1<A>) -> Array2<A> {
    // TODO find more idiomatic code
    // let mut out = Array::zeros((a.len(), b.len()));
    let mut out = b
        .insert_axis(Axis(0))
        .broadcast((a.len(), b.len()))
        .expect("the b array can be broadcast")
        .to_owned();
    // let mut out = ndarray::stack(Axis(0), vec![b.broadcast(()); a.len()].as_slice()).unwrap();
    out *= &a.insert_axis(Axis(1));
    // for i in 0..a.len() {
    //     out.slice_mut(s![i, ..]) *= a;
    //     // out.slice_mut(s![i, ..]).assign(b * a[i]);
    //     // for j in 0..b.len() {
    //     //     out[[i, j]] = a[i] * b[j];
    //     // }
    // }
    out
}

#[cfg(test)]
speculate! {
    describe "fn outer()" {
        it "calculates the outer product of 2x3 array" {
            let a = array![-1., 1.];
            let b = array![1., 2., 3.];
            let expected = array![[-1., -2., -3.],
                                  [1., 2., 3.]];
            assert_eq!(outer(a.view(), b.view()), expected);
        }

        it "calculates the outer product of 2x2 array" {
            assert_eq!(outer(array![-1., 1.].view(),
                             array![3., 7.].view()),
                       array![[-3., -7.,],
                              [3., 7.]]);
        }
    }
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
