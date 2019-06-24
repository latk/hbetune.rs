use ndarray::prelude::*;

use crate::kernel::Scalar;

#[derive(Debug)]
pub struct Space {
    params: Vec<Parameter>,
}

impl Space {
    pub fn new() -> Self {
        let params = Vec::new();
        Space { params }
    }

    pub fn len(&self) -> usize { self.params.len() }

    pub fn add_real_parameter(&mut self, name: impl Into<String>, lo: f64, hi: f64) {
        assert!(lo < hi);
        self.params.push(Parameter::Real { name: name.into(), lo, hi });
    }

    pub fn into_transformed_mut<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            *x = match *param {
                Parameter::Real { name: _, lo, hi } => (*x - A::from_f(lo)) / A::from_f(hi - lo),
            };
        }
    }

    pub fn into_transformed<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.into_transformed_mut(x.view_mut());
        x
    }

    pub fn into_transformed_a<A: Scalar>(&self, mut xs: Array2<A>) -> Array2<A> {
        for x in xs.outer_iter_mut() {
            self.into_transformed_mut(x);
        }
        xs
    }

    pub fn from_transformed_mut<A: Scalar>(&self, mut x: ArrayViewMut1<A>) {
        assert!(x.len() == self.len());
        for (x, param) in x.iter_mut().zip(&self.params) {
            *x = match *param {
                Parameter::Real { name: _, lo, hi } => *x * A::from_f(hi - lo) + A::from_f(lo),
            };
        }
        unimplemented!()
    }

    pub fn from_transformed<A: Scalar>(&self, mut x: Array1<A>) -> Array1<A> {
        self.from_transformed_mut(x.view_mut());
        x
    }
}

#[derive(Debug)]
enum Parameter {
    Real { name: String, lo: f64, hi: f64 },
}
