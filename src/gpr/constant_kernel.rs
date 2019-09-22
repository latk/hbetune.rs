use ndarray::prelude::*;

use crate::gpr::{Kernel, Scalar};
use crate::util::{BoundedValue, BoundsError};

/// A constant kernel.
/// Can be used to scale the magnitude of another kernel.
#[derive(Clone, Debug)]
pub struct ConstantKernel {
    constant: BoundedValue<f64>,
}

impl ConstantKernel {
    pub fn new(constant: BoundedValue<f64>) -> Self {
        ConstantKernel { constant }
    }

    pub fn constant(&self) -> BoundedValue<f64> {
        self.constant.clone()
    }
}

impl Kernel for ConstantKernel {
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A> {
        Array::from_elem(
            (x1.shape()[0], x2.shape()[0]),
            A::from_f(self.constant.value()),
        )
    }

    fn theta_grad<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let kernel = self.kernel(x, x);
        let gradient = Array::from_elem(
            (x.shape()[0], x.shape()[0], 1),
            A::from_f(self.constant.value()),
        );
        (kernel, gradient)
    }

    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A> {
        Array::from_elem(x.shape()[0], A::from_f(self.constant.value()))
    }

    fn n_params(&self) -> usize {
        1
    }

    fn theta(&self) -> Vec<f64> {
        vec![self.constant.value().ln()]
    }

    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>> {
        let constant = unpack_theta_one(theta).expect("theta slice must contain exactly one value");
        let constant = self.constant.with_value(constant.exp())?;
        Ok(Self::new(constant))
    }

    fn with_clamped_theta(self, theta: &[f64]) -> Self {
        let constant = unpack_theta_one(theta).expect("theta slice must contain exactly one value");
        let constant = self.constant.with_clamped_value(constant.exp());
        Self::new(constant)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(self.constant.min().ln(), self.constant.max().ln())]
    }
}

fn unpack_theta_one(theta: &[f64]) -> Option<f64> {
    match *theta {
        [value] => Some(value),
        _ => None,
    }
}
