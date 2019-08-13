use noisy_float::prelude::*;

pub trait Scalar
    : ndarray::NdFloat
    + ndarray::ScalarOperand
    + ndarray_linalg::lapack::Lapack
    + num_traits::float::Float
    + From<i16>  // for literals
    + From<f32>  // for literals
    // + From<f64>
    + Into<f64>
    + rand::distributions::uniform::SampleUniform
    + float_cmp::ApproxEq
{
    fn from_f<F: Into<f64>>(x: F) -> Self;
    fn from_i(x: i16) -> Self { std::convert::From::from(x) }
    fn to_n64(self) -> N64 { n64(self.into()) }
}

impl Scalar for f64 {
    fn from_f<F: Into<f64>>(x: F) -> Self {
        x.into()
    }
}

impl Scalar for f32 {
    fn from_f<F: Into<f64>>(x: F) -> Self {
        x.into() as f32
    }
}
