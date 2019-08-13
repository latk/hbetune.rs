use itertools::Itertools as _;
use ndarray::prelude::*;

use crate::gpr::{Kernel, Scalar};

pub fn predict<A>(
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
                elements_below_threshold.into_iter().format(", "),
            );
        }

        variance_out.assign(&y_var);
    }

    y
}

#[test]
fn it_works_on_a_simple_case() {
    use crate::gpr::{ConstantKernel, FittedKernel, Matern, Product};
    use crate::util::BoundedValue;

    // problem definition
    let xs = array![[0.0], [0.5], [0.5], [1.0]];
    let ys = array![0.0, 0.8, 1.2, 2.0];
    let kernel = Product::of(
        ConstantKernel::new(BoundedValue::new(3.0, 0.1, 4.0).unwrap()),
        Matern::new(2.5, vec![BoundedValue::new(1.5, 0.1, 2.0).unwrap()]),
    );
    const SEED: usize = 938274;
    const N_RESTARTS_OPTIMIZER: usize = 4;
    let mut rng = crate::RNG::new_with_seed(SEED);

    // precompute stuff
    let FittedKernel {
        kernel,
        alpha,
        k_inv,
        ..
    } = FittedKernel::new(
        kernel,
        xs.clone(),
        ys.clone(),
        &mut rng,
        N_RESTARTS_OPTIMIZER,
        BoundedValue::new(1.0, 0.001, 1.0).unwrap(),
    );

    // perform prediction
    let predict_xs = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
    let mut variances = Array1::zeros(5);
    let prediction = predict(
        &kernel,
        alpha.view(),
        predict_xs.view(),
        xs.view(),
        k_inv.view(),
        Some(variances.view_mut()),
    );

    assert_all_close!(prediction, array![0.0, 0.5, 1.0, 1.5, 2.0], 0.1);
    assert_all_close!(variances, Array1::from_elem(5, 0.03), 0.03);
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
speculate::speculate! {
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
