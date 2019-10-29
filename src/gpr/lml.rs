use ndarray::prelude::*;
use ndarray_linalg::CholeskyFactorized;
use num_traits::Float;

use crate::gpr::{Kernel, Scalar};

/// Calculate the log marginal likelihood and gradients of a certain kernel/noise configuration.
pub struct LmlWithGradient<A> {
    pub lml: f64,
    pub lml_gradient: Vec<f64>,
    pub alpha: Array1<A>,
    pub factorization: CholeskyFactorized<ndarray::OwnedRepr<A>>,
}

impl<A> LmlWithGradient<A> {
    pub fn of(
        kernel: impl Kernel,
        noise: A,
        x_train: ArrayView2<A>,
        y_train: ArrayView1<A>,
    ) -> Option<Self>
    where
        A: Scalar,
    {
        lml_with_gradient(kernel, noise, x_train, y_train)
    }
}

fn lml_with_gradient<A: Scalar>(
    kernel: impl Kernel,
    noise: A,
    x_train: ArrayView2<A>,
    y_train: ArrayView1<A>,
) -> Option<LmlWithGradient<A>> {
    use ndarray_linalg::cholesky::*;

    // calculate combined gradient which has dim3
    // stack(noise gradient, kernel gradient).
    // However, actually performing the stacking is unnecessary.
    let (mut kernel_matrix, kernel_gradient) = kernel.theta_grad(x_train);
    let noise_gradient = Array2::eye(x_train.nrows()) * noise;

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
        let tmp = outer::outer(alpha.view(), alpha.view()) - &factorization.invc().unwrap();
        // Compute "0.5 * trace(tmp dot gradient)"
        // without constructing the full matrix
        // as only the diagonal is required.
        // Additionally, we can split up the computation of the stacked gradient matrix
        std::iter::once(noise_gradient.view())
            .chain(kernel_gradient.axis_iter(Axis(2)))
            .map(|grad| 0.5 * (&tmp * &grad).sum().into())
            .collect()
    };

    Some(LmlWithGradient {
        lml,
        lml_gradient,
        alpha,
        factorization,
    })
}

mod outer {
    use crate::gpr::Scalar;
    use ndarray::prelude::*;

    pub fn outer<A: Scalar>(a: ArrayView1<A>, b: ArrayView1<A>) -> Array2<A> {
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

    #[test]
    fn it_calculates_the_outer_product_of_2_by_3_array() {
        let a = array![-1., 1.];
        let b = array![1., 2., 3.];
        let expected = array![[-1., -2., -3.], [1., 2., 3.]];
        assert_eq!(outer(a.view(), b.view()), expected);
    }

    #[test]
    fn it_calculates_the_outer_product_of_2_by_2_array() {
        assert_eq!(
            outer(array![-1., 1.].view(), array![3., 7.].view()),
            array![[-3., -7.,], [3., 7.]]
        );
    }
}
