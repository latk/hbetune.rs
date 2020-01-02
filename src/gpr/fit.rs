use crate::gpr::{Kernel, LmlWithGradient, Scalar};
use crate::util::BoundedValue;
use crate::RNG;
use ndarray::prelude::*;

pub struct FittedKernel<K, A> {
    pub kernel: K,
    pub noise: BoundedValue<f64>,
    pub alpha: Array1<A>,
    pub k_inv: Array2<A>,
    pub lml: f64,
}

impl<K, A> FittedKernel<K, A> {
    /// Assign kernel parameters with maximal log-marginal likelihood.
    ///
    /// The current kernel parameters (theta) are used as an optimization starting point.
    pub fn new(
        kernel: K,
        x_train: Array2<A>,
        y_train: Array1<A>,
        rng: &mut RNG,
        n_restarts_optimizer: usize,
        noise: BoundedValue<f64>,
    ) -> Self
    where
        A: Scalar,
        K: Kernel,
    {
        fit_kernel(kernel, x_train, y_train, rng, n_restarts_optimizer, noise)
    }

    pub fn extend(
        kernel: K,
        x_train: Array2<A>,
        y_train: Array1<A>,
        noise: BoundedValue<f64>,
    ) -> Self
    where
        A: Scalar,
        K: Kernel,
    {
        let LmlWithGradient {
            lml,
            lml_gradient: _lml_gradient,
            alpha,
            factorization,
        } = match LmlWithGradient::of(
            kernel.clone(),
            A::from_f(noise.value()),
            x_train.view(),
            y_train.view(),
        ) {
            Some(result) => result,
            None => panic!("Kernel matrix must be invertible."),
        };

        // Precompute arrays needed at prediction.
        use ndarray_linalg::cholesky::*;
        let k_inv = factorization.invc_into().unwrap();
        FittedKernel {
            kernel,
            noise,
            alpha,
            k_inv,
            lml,
        }
    }
}

fn fit_kernel<K: Kernel, A: Scalar>(
    mut kernel: K,
    x_train: Array2<A>,
    y_train: Array1<A>,
    rng: &mut RNG,
    n_restarts_optimizer: usize,
    noise: BoundedValue<f64>,
) -> FittedKernel<K, A> {
    let n_observations = x_train.nrows();
    assert!(y_train.dim() == n_observations);

    struct Capture<A: Scalar> {
        lml: f64,
        theta: Vec<f64>,
        noise: f64,
        mat_k_factorization: ndarray_linalg::CholeskyFactorized<ndarray::OwnedRepr<A>>,
        alpha: Array1<A>,
    };

    // calculate the negative log-marginal likelihood of a theta-vector
    // and its gradients
    let capture: std::cell::RefCell<Option<Capture<A>>> = None.into();
    let obj_func = |theta: &[f64], want_gradient: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let (noise_theta, kernel_theta) = theta.split_first().unwrap();
        let kernel = kernel.clone().with_clamped_theta(kernel_theta);
        let noise: A = A::from_f(noise_theta.exp());

        let LmlWithGradient {
            lml,
            lml_gradient,
            alpha,
            factorization,
        } = match LmlWithGradient::of(kernel, noise, x_train.view(), y_train.view()) {
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

    kernel = kernel.clone().with_clamped_theta(theta.as_slice());
    let noise = noise.with_clamped_value(captured_noise);

    // Precompute arrays needed at prediction
    use ndarray_linalg::cholesky::*;
    let mat_k_inv = mat_k_factorization.invc_into().unwrap();
    FittedKernel {
        kernel,
        noise,
        alpha,
        k_inv: mat_k_inv,
        lml,
    }
}
