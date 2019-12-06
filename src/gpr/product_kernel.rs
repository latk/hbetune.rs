use ndarray::prelude::*;

use crate::gpr::{Kernel, Scalar};
use crate::util::BoundsError;

/// A product of two kernels `k1 * k2`
#[derive(Clone)]
pub struct Product<K1, K2>
where
    K1: Clone + Kernel,
    K2: Clone + Kernel,
{
    k1: K1,
    k2: K2,
}

impl<K1: Clone + Kernel, K2: Clone + Kernel> Product<K1, K2> {
    pub fn of(k1: K1, k2: K2) -> Self {
        Product { k1, k2 }
    }

    pub fn k1(&self) -> &K1 {
        &self.k1
    }

    pub fn k2(&self) -> &K2 {
        &self.k2
    }
}

impl<K1, K2> Kernel for Product<K1, K2>
where
    K1: Kernel,
    K2: Kernel,
{
    fn kernel<A: Scalar>(&self, xa: ArrayView2<A>, xb: ArrayView2<A>) -> Array2<A> {
        self.k1.kernel(xa, xb) * self.k2.kernel(xa, xb)
    }

    fn theta_grad<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let (kernel1, gradient1) = self.k1.theta_grad(x);
        let (kernel2, gradient2) = self.k2.theta_grad(x);

        let dim0 = kernel1.nrows();
        let dim1 = kernel1.ncols();
        let dim2a = gradient1.len_of(Axis(2));
        let dim2b = gradient2.len_of(Axis(2));

        assert_eq!(dim0, kernel2.nrows());
        assert_eq!(dim0, gradient1.len_of(Axis(0)));
        assert_eq!(dim0, gradient2.len_of(Axis(0)));
        assert_eq!(dim1, kernel2.ncols());
        assert_eq!(dim1, gradient1.len_of(Axis(1)));
        assert_eq!(dim1, gradient2.len_of(Axis(1)));

        let kernel = &kernel1 * &kernel2;
        let g1k2 = gradient1 * kernel2.insert_axis(Axis(2));
        let g2k1 = gradient2 * kernel1.insert_axis(Axis(2));

        // SAFETY: shape bounds guarantee correct accesses
        let gradient = Array::from_shape_fn((dim0, dim1, dim2a + dim2b), |(i0, i1, i2)| {
            if i2 < dim2a {
                *unsafe { g1k2.uget([i0, i1, i2]) }
            } else {
                *unsafe { g2k1.uget([i0, i1, i2 - dim2a])}
            }
        });

        (kernel, gradient)
    }

    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A> {
        self.k1.diag(x) * self.k2.diag(x)
    }

    fn n_params(&self) -> usize {
        self.k1.n_params() + self.k2.n_params()
    }

    fn theta(&self) -> Vec<f64> {
        let mut theta = Vec::new();
        theta.extend(self.k1.theta());
        theta.extend(self.k2.theta());
        theta
    }

    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>> {
        assert!(theta.len() == self.n_params());
        let (theta1, theta2) = theta.split_at(self.k1.n_params());
        let k1 = self.k1.with_theta(theta1)?;
        let k2 = self.k2.with_theta(theta2)?;
        Ok(Self::of(k1, k2))
    }

    fn with_clamped_theta(self, theta: &[f64]) -> Self {
        assert_eq!(theta.len(), self.n_params());
        let (theta1, theta2) = theta.split_at(self.k1.n_params());
        let k1 = self.k1.with_clamped_theta(theta1);
        let k2 = self.k2.with_clamped_theta(theta2);
        Self::of(k1, k2)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = Vec::new();
        bounds.extend(self.k1.bounds());
        bounds.extend(self.k2.bounds());
        bounds
    }
}

impl<K1: Kernel, K2: Kernel> std::fmt::Debug for Product<K1, K2> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("Product")
            .field(&self.k1)
            .field(&self.k2)
            .finish()
    }
}

#[test]
#[allow(clippy::unreadable_literal)]
fn it_produces_a_kernel_and_gradient() {
    use crate::gpr::{ConstantKernel, Matern};
    use crate::util::BoundedValue;

    let kernel = Product::of(
        ConstantKernel::new(BoundedValue::new(2.0, 1.0, 5.0).unwrap()),
        Matern::new(
            2.5,
            vec![
                BoundedValue::new(1.0, 0.05, 20.0).unwrap(),
                BoundedValue::new(1.0, 0.05, 20.0).unwrap(),
            ],
        ),
    );

    let x = array![[0.5, 7.8], [3.3, 1.4], [3.9, 5.6],];

    // produced by sklearn
    let kernel_matrix = array![
        [2.00000000e+00, 3.22221679e-05, 8.73105609e-03],
        [3.22221679e-05, 2.00000000e+00, 6.14136045e-03],
        [8.73105609e-03, 6.14136045e-03, 2.00000000e+00]
    ];

    // produced by sklearn
    let gradient_matrix = array![
        [
            [2.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [3.22221679e-05, 7.14401245e-05, 3.73238201e-04],
            [8.73105609e-03, 4.52409267e-02, 1.89417029e-02]
        ],
        [
            [3.22221679e-05, 7.14401245e-05, 3.73238201e-04],
            [2.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [6.14136045e-03, 9.54435058e-04, 4.67673178e-02]
        ],
        [
            [8.73105609e-03, 4.52409267e-02, 1.89417029e-02],
            [6.14136045e-03, 9.54435058e-04, 4.67673178e-02],
            [2.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        ]
    ];

    let (actual_kernel, actual_gradient) = kernel.theta_grad(x.view());
    assert_all_close!(&actual_kernel, &kernel_matrix, 1e-3);
    assert_all_close!(actual_gradient, gradient_matrix, 1e-3);
    assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
}
