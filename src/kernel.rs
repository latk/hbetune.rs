//! Adapted from the sklearn.gaussian_process.kernels Python module.

use ndarray::prelude::*;
use num_traits::Float;

/// A kernel function that can be used in a Gaussian Process
/// and can be tuned.
pub trait Kernel: Clone + std::fmt::Debug {
    /// Evaluate the kernel function.
    /// Input arrays have shape (n_samples, n_features).
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A>;

    /// Get the gradients of the kernel function in a certain position.
    /// Input array has shape (n_samples, n_features).
    fn gradient<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>);

    /// Get the diagonal of the kernel function in a certain position.
    /// Input array has shape (n_samples, n_features).
    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A>;

    /// Number of tunable parameters.
    fn n_params(&self) -> usize;

    /// Get the log-transformed parameters, useful for tuning.
    fn theta(&self) -> Vec<f64>;

    // Update the kernel parameters with a theta vector.
    // Returns None if the theta is unacceptable, e.g. because it violates bounds.
    // Panics if the theta has wrong length – must equal n_params().
    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>>
    where Self: Sized;

    /// Update the kernel parameters with a theta vector.
    /// Adjust the parameters if they would otherwise violate bounds.
    /// Panics if the theta vector has wrong lenght – must equal n_params().
    fn with_clamped_theta(self, theta: &[f64]) -> Self
    where Self: Sized;

    // Get the theta bounds for tuning (log-transformed).
    fn bounds(&self) -> Vec<(f64, f64)>;
}

/// A value with some bounds for tuning.
#[derive(Clone, PartialEq)]
pub struct BoundedValue<A: PartialOrd + Clone> {
    value: A,
    max: A,
    min: A
}

impl<A: PartialOrd + Clone> BoundedValue<A> where A: PartialEq + Copy {
    /// Create a new value with bounds (inclusive).
    pub fn new(value: A, min: A, max: A) -> Result<Self, BoundsError<A>> {
        if min <= value && value <= max {
            Ok(BoundedValue { value, min, max })
        } else {
            Err(BoundsError { value, min, max })
        }
    }

    /// Get the current value.
    pub fn value(&self) -> A { self.value }

    /// Get the lower bound.
    pub fn min(&self) -> A { self.min }

    /// Get the upper bound.
    pub fn max(&self) -> A { self.max }

    /// Set a new value.
    pub fn with_value(self, value: A) -> Result<Self, BoundsError<A>> {
        Self::new(value, self.min, self.max)
    }

    /// Set a new value. If bounds are violated, substitute the bound instead.
    pub fn with_clamped_value(self, value: A) -> Self {
        let value =
            if value < self.min {
                self.min
            } else if self.max < value {
                self.max
            } else {
                value
            };
        Self { value, min: self.min, max: self.max }
    }
}

impl<A> std::fmt::Debug for BoundedValue<A>
where A: PartialOrd + Clone + std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BoundedValue({:?} in {:?} .. {:?})", self.value, self.min, self.max)
    }
}

#[derive(Debug)]
pub struct BoundsError<A> {
    value: A,
    min: A,
    max: A,
}

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

    pub fn constant(&self) -> BoundedValue<f64> { self.constant.clone() }
}

impl Kernel for ConstantKernel {
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A> {
        Array::from_elem((x1.shape()[0], x2.shape()[0]),
                         self.constant.value().into_scalar())
    }

    fn gradient<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let kernel = self.kernel(x, x);
        let gradient = Array::from_elem((x.shape()[0], x.shape()[0], 1),
                                        self.constant.value().into_scalar());
        (kernel, gradient)
    }

    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A> {
        Array::from_elem(x.shape()[0],
                         self.constant.value().into_scalar())
    }

    fn n_params(&self) -> usize { 1 }

    fn theta(&self) -> Vec<f64> {
        vec![self.constant.value().ln()]
    }

    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>> {
        let constant = unpack_theta_one(theta)
            .expect("theta slice must contain exactly one value");
        let constant = self.constant.with_value(constant.exp())?;
        Ok(Self::new(constant))
    }

    fn with_clamped_theta(self, theta: &[f64]) -> Self {
        let constant = unpack_theta_one(theta)
            .expect("theta slice must contain exactly one value");
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

/// Matern kernel family, parameterized by smoothness parameter nu.
/// Allows separate length scales so that it can be used as an anisotropic kernel.
/// Nu should be of for p/2, e.g. 3/2.
#[derive(Clone, Debug)]
pub struct Matern {
    nu: f64,
    length_scale: Vec<BoundedValue<f64>>,
}

impl Matern {
    /// Create a new matern kernel with certain smoothness and length scales.
    pub fn new(nu: f64, length_scale: Vec<BoundedValue<f64>>) -> Self {
        Matern { nu, length_scale }
    }

    pub fn nu(&self) -> f64 { self.nu }

    pub fn length_scale(&self) -> &[BoundedValue<f64>] { self.length_scale.as_slice() }

    fn length_scale_array(&self) -> Array1<f64> {
        self.length_scale.iter().map(BoundedValue::value).collect()
    }
}

impl Kernel for Matern {
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A> {
        assert_eq!(x1.cols(), self.n_params(),
                   "number of x1 columns must match number of features");
        assert_eq!(x2.cols(), self.n_params(),
                   "number of x2 columns must match number of features");

        // normalize the arrays via their length scale
        let length_scale = self.length_scale_array().mapv(A::from_f);
        let normalize_length_scale = |xs: ArrayView2<A>| {
            xs.to_owned() / length_scale.view().insert_axis(Axis(0)).broadcast(xs.shape()).unwrap()
        };
        let x1 = normalize_length_scale(x1);
        let x2 = normalize_length_scale(x2);

        // raw euclidean distance matrix
        let dists = cdist(x1.view(), x2.view());

        let kernel = match self.nu {
            nu if nu == 0.5 => (-dists).exp(),
            nu if nu == 1.5 => {
                let kernel: Array2<A> = dists * A::from_f(3.0.sqrt());
                (kernel.clone() + A::from_i(1))
                    * (-kernel).exp()
            },
            nu if nu == 2.5 => {
                // K = dists * sqrt(4)
                let kernel: Array2<A> = dists * A::from_f(5.0.sqrt());
                // (1 + K + K**2 / 3) * exp(-K)
                (kernel.clone().powi(2) / A::from_i(3) + &kernel + A::from_i(1))
                    * (-kernel).exp()
            },
            _ => unimplemented!("Matern kernel with arbitrary values for nu"),
        };
        kernel
    }

    fn gradient<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let kernel = self.kernel(x, x);
        let gradient_shape = (x.rows(), x.rows(), self.n_params());
        let length_scale = self.length_scale_array().mapv(A::from_f);

        // pairwise dimension-wise distances: d[ijk] = (x[ik] - x[jk])**2 / length_scale[k]**2
        let d_shape = (x.rows(), x.rows(), x.cols());
        let d = (&x.insert_axis(Axis(1)).broadcast(d_shape).unwrap()
                 - &x.insert_axis(Axis(0)).broadcast(d_shape).unwrap()).powi(2)
            / length_scale.powi(2).insert_axis(Axis(0)).insert_axis(Axis(0));
        assert_eq!(d.shape(), &[x.rows(), x.rows(), self.n_params()]);

        let gradient = match self.nu {
            nu if nu == 0.5 => {
                let mut gradient = kernel.clone().insert_axis(Axis(2)) * &d
                    / d.sum_axis(Axis(2)).sqrt().insert_axis(Axis(2));
                gradient.map_inplace(|x| if !x.is_finite() { *x = 0f32.into() });
                gradient
            },
            nu if nu == 1.5 => {
                // gradient = 3 * d * exp(-sqrt(3 * d.sum(-1)))[..., np.newaxis]
                let tmp = (d.sum_axis(Axis(2)) * A::from_i(3)).sqrt();
                let gradient = &d.broadcast(gradient_shape).unwrap()
                    * &(-tmp).exp().insert_axis(Axis(2)).broadcast(gradient_shape).unwrap()
                    * A::from_i(3);
                gradient
            },
            nu if nu == 2.5 => {
                let tmp: Array3<A> = (d.sum_axis(Axis(2)) * A::from_i(5)).sqrt().insert_axis(Axis(2));
                let gradient = (-tmp.clone()).exp().broadcast(gradient_shape).unwrap().to_owned()
                    * &(tmp + A::from_i(1)).broadcast(gradient_shape).unwrap()
                    * d.broadcast(gradient_shape).unwrap()
                    * A::from_f(5./3.);
                gradient
            },
            _ => unimplemented!("Matern kernel gradient with arbitrary values for nu"),
        };
        (kernel, gradient)
    }

    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A> {
        Array::ones(x.rows())
    }

    fn n_params(&self) -> usize {
        self.length_scale.len()
    }

    fn theta(&self) -> Vec<f64> {
        self.length_scale.iter().map(BoundedValue::value).map(f64::ln).collect()
    }

    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>> {
        assert!(theta.len() == self.n_params());

        let length_scale = theta.iter().cloned()
            .map(f64::exp)
            .zip(self.length_scale)
            .map(|(value, bounded)| bounded.with_value(value))
            .collect::<Result<_, _>>()?;

        Ok(Self::new(self.nu, length_scale))
    }

    fn with_clamped_theta(self, theta: &[f64]) -> Self {
        assert!(theta.len() == self.n_params());

        let length_scale = theta.iter().cloned()
            .map(f64::exp)
            .zip(self.length_scale)
            .map(|(value, bounded)| bounded.with_clamped_value(value))
            .collect::<Vec<_>>();

        Self::new(self.nu, length_scale)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.length_scale.iter().map(|bounds| (bounds.min().ln(), bounds.max().ln())).collect()
    }
}

#[cfg(test)] speculate::speculate! {

    describe "Matern kernel" {

        it "produces a kernel and gradient with nu = 3/2" {
            let kernel = Matern::new(
                1.5,
                vec![
                    BoundedValue::new(1., 0.05, 20.).unwrap(),
                    BoundedValue::new(1., 0.05, 20.).unwrap(),
                ]);

            let x = array![
                [0., 0.],
                [1., 1.],
                [1., 2.]];

            let kernel_matrix = array![  // produced by sklearn
                [ 1.        ,  0.29782077,  0.1013397 ],
                [ 0.29782077,  1.        ,  0.48335772],
                [ 0.1013397 ,  0.48335772,  1.        ]];

            let gradient_matrix = array![  // produced by sklearn
                [[ 0.        ,  0.        ],
                 [ 0.25901289,  0.25901289],
                 [ 0.0623887 ,  0.24955481]],
                [[ 0.25901289,  0.25901289],
                 [ 0.        ,  0.        ],
                 [ 0.        ,  0.53076362]],
                [[ 0.0623887 ,  0.24955481],
                 [ 0.        ,  0.53076362],
                 [ 0.        ,  0.        ]]];

            let (actual_kernel, actual_gradient) = kernel.gradient(x.view());
            assert_all_close!(&actual_kernel, &kernel_matrix, 0.001);
            assert_all_close!(actual_gradient, gradient_matrix, 0.001);
            assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
        }

        it "produces a kernel and gradient with nu = 5/2" {
            let kernel = Matern::new(
                2.5,
                vec![
                    BoundedValue::new(1., 0.05, 20.).unwrap(),
                    BoundedValue::new(1., 0.05, 20.).unwrap(),
                ]);

            let x = array![
                [0., 0.],
                [1., 1.],
                [1., 2.]];

            let kernel_matrix = array![  // produced by sklearn
                [ 1.        ,  0.31728336,  0.09657724],
                [ 0.31728336,  1.        ,  0.52399411],
                [ 0.09657724,  0.52399411,  1.        ]];

            let gradient_matrix = array![  // produced by sklearn
                [[ 0.        ,  0.        ],
                 [ 0.29364328,  0.29364328],
                 [ 0.06737947,  0.26951788]],
                [[ 0.29364328,  0.29364328],
                 [ 0.        ,  0.        ],
                 [ 0.        ,  0.57644039]],
                [[ 0.06737947,  0.26951788],
                 [ 0.        ,  0.57644039],
                 [ 0.        ,  0.        ]]];

            let (actual_kernel, actual_gradient) = kernel.gradient(x.view());
            assert_all_close!(&actual_kernel, &kernel_matrix, 0.001);
            assert_all_close!(actual_gradient, gradient_matrix, 0.001);
            assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
        }

    }

    describe "Product kernel" {
        it "produces a kernel and gradient" {
            let kernel = Product::of(
                ConstantKernel::new(
                    BoundedValue::new(2.0, 1.0, 5.0).unwrap(),
                ),
                Matern::new(
                    2.5,
                    vec![
                        BoundedValue::new(1.0, 0.05, 20.0).unwrap(),
                        BoundedValue::new(1.0, 0.05, 20.0).unwrap(),
                    ],
                ),
            );

            let x = array![
                [0.5, 7.8],
                [3.3, 1.4],
                [3.9, 5.6],
            ];

            // produced by sklearn
            let kernel_matrix = array![[  2.00000000e+00,   3.22221679e-05,   8.73105609e-03],
                                       [  3.22221679e-05,   2.00000000e+00,   6.14136045e-03],
                                       [  8.73105609e-03,   6.14136045e-03,   2.00000000e+00]];

            // produced by sklearn
            let gradient_matrix = array![[[  2.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                          [  3.22221679e-05,   7.14401245e-05,   3.73238201e-04],
                                          [  8.73105609e-03,   4.52409267e-02,   1.89417029e-02]],
                                         [[  3.22221679e-05,   7.14401245e-05,   3.73238201e-04],
                                          [  2.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                                          [  6.14136045e-03,   9.54435058e-04,   4.67673178e-02]],
                                         [[  8.73105609e-03,   4.52409267e-02,   1.89417029e-02],
                                          [  6.14136045e-03,   9.54435058e-04,   4.67673178e-02],
                                          [  2.00000000e+00,   0.00000000e+00,   0.00000000e+00]]];

            let (actual_kernel, actual_gradient) = kernel.gradient(x.view());
            assert_all_close!(&actual_kernel, &kernel_matrix, 1e-3);
            assert_all_close!(actual_gradient, gradient_matrix, 1e-3);
            assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
        }
    }

}

/// A product of two kernels `k1 * k2`
#[derive(Clone)]
pub struct Product<K1, K2>
where K1: Clone + Kernel,
      K2: Clone + Kernel,
{
    k1: K1,
    k2: K2,
}

impl<K1: Clone + Kernel, K2: Clone + Kernel> Product<K1, K2> {
    pub fn of(k1: K1, k2: K2) -> Self { Product { k1, k2 }}

    pub fn k1(&self) -> &K1 { &self.k1 }

    pub fn k2(&self) -> &K2 { &self.k2 }
}

impl<K1, K2> Kernel for Product<K1, K2>
where K1: Kernel,
      K2: Kernel,
{
    fn kernel<A: Scalar>(&self, xa: ArrayView2<A>, xb: ArrayView2<A>) -> Array2<A> {
        self.k1.kernel(xa, xb) * self.k2.kernel(xa, xb)
    }

    fn gradient<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let (kernel1, gradient1) = self.k1.gradient(x);
        let (kernel2, gradient2) = self.k2.gradient(x);
        let kernel = &kernel1 * &kernel2;
        let gradient = stack!(Axis(2),
                              gradient1 * kernel2.insert_axis(Axis(2)),
                              gradient2 * kernel1.insert_axis(Axis(2)));
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

fn cdist<A: Scalar>(xa: ArrayView2<A>, xb: ArrayView2<A>) -> Array2<A> {
    let mut out = Array2::zeros((xa.rows(), xb.rows()));
    for (i, a) in xa.outer_iter().enumerate() {
        for (j, b) in xb.outer_iter().enumerate() {
            out[[i, j]] = (&a - &b).mapv_into(|x| x.powi(2)).sum().sqrt();
        }
    };
    out
}

#[cfg(test)] speculate::speculate! {
    describe "fn cdist()" {
        it "calculates single distance" {
            let a = array![[1., 3.]];
            let b = array![[2., 5.]];
            // (1 - 2)**2 + (3 - 5)**2 = 1**2 + 2**2 = 5
            assert_eq!(cdist(a.view(), b.view()), array![[5.0.sqrt()]])
        }

        it "calculates multiple distances" {
            let a = array![[0., 0.],
                           [1.,1.],
                           [2., 2.]];
            let b = array![[1., 2.],
                           [3., 4.]];
            assert_eq!(cdist(a.view(), b.view()),
                       array![[5.0.sqrt(), 25.0.sqrt()],
                              [1.0.sqrt(), 13.0.sqrt()],
                              [1.0.sqrt(), 5.0.sqrt()]]);
        }
    }
}

pub trait Scalar
    : NdFloat
    + ndarray::ScalarOperand
    + ndarray_linalg::lapack::Lapack
    + num_traits::float::Float
    + From<i16>  // for literals
    + From<f32>  // for literals
    // + From<f64>
    + Into<f64>
    + rand::distributions::uniform::SampleUniform
{
    fn from_f<F: Into<f64>>(x: F) -> Self;
    fn from_i(x: i16) -> Self { std::convert::From::from(x) }
}

impl Scalar for f64 {
    fn from_f<F: Into<f64>>(x: F) -> Self { x.into() }
}

impl Scalar for f32 {
    fn from_f<F: Into<f64>>(x: F) -> Self { x.into() as f32 }
}

pub trait IntoScalar {
    fn into_scalar<S: Scalar>(self) -> S;
}

impl<T: num_traits::float::Float + Scalar> IntoScalar for T {
    fn into_scalar<S: Scalar>(self) -> S { Scalar::from_f(self) }
}

trait ArrayExt<A: Scalar, D: Dimension> {
    fn powf(self, exponent: A) -> Array<A, D>;
    fn powi(self, exponent: i32) -> Array<A, D>;
    fn exp(self) -> Array<A, D>;
    fn sqrt(self) -> Array<A, D>;
}

impl<A: Scalar, S, D> ArrayExt<A, D> for ArrayBase<S, D>
where S: ndarray::Data<Elem = A>,
      D: Dimension,
{
    fn powf(self, exponent: A) -> Array<A, D> {
        self.into_owned().mapv_into(|x| x.powf(exponent))
    }

    fn powi(self, exponent: i32) -> Array<A, D> {
        self.into_owned().mapv_into(|x| x.powi(exponent))
    }

    fn exp(self) -> Array<A, D> {
        self.into_owned().mapv_into(num_traits::Float::exp)
    }

    fn sqrt(self) -> Array<A, D> {
        self.into_owned().mapv_into(num_traits::Float::sqrt)
    }
}
