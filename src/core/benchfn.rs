use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{Float, FloatConst};

/// Sphere function: N-dimensional, symmetric.
///
/// Bounds: unbounded, but -2 <= xi <= 2 is sensible.
///
/// Optimum: f(0, ..., 0) = 0
///
/// ```
/// # #[macro_use] extern crate ndarray;
/// # use ggtune::benchfn::*;
/// assert_eq!(sphere(array![0.0]), 0.0);
/// assert_eq!(sphere(array![0.0, 0.0]), 0.0);
/// assert_eq!(sphere(array![1.0, 2.0]), 5.0);
/// assert_eq!(sphere(array![0.0, 0.0, 0.0, 0.0, 0.0]), 0.0);
/// ```
pub fn sphere<A, S>(xs: ArrayBase<S, Ix1>) -> A
where
    S: Data<Elem = A>,
    A: Clone + Float,
{
    assert!(!xs.is_empty(), "at least one dimension required");
    xs.mapv(|x| x.powi(2)).sum()
}

/// Goldstein-Price: Asymmetric function with single optimum.
///
/// Bounds: -2 <= xi <= 2
///
/// Optimum: f(0, -1) = 3
///
/// ```
/// # use ggtune::benchfn::goldstein_price;
/// assert_eq!(goldstein_price(0.0f64, -1.0), 3.0);
/// assert_eq!(goldstein_price(0.0f32, -1.0), 3.0);
/// ```
///
/// Definition taken from:
/// https://en.wikipedia.org/wiki/Test_functions_for_optimization
pub fn goldstein_price<A>(x1: A, x2: A) -> A
where
    A: Float,
    u16: Into<A>,
{
    let a = |x: u16| x.into();

    (a(1)
        + (x1 + x2 + a(1)).powi(2)
            * (a(19) - a(14) * x1 + a(3) * x1.powi(2) - a(14) * x2
                + a(6) * x1 * x2
                + a(3) * x2.powi(2)))
        * (a(30)
            + (a(2) * x1 - a(3) * x2).powi(2)
                * (a(18) - a(32) * x1 + a(12) * x1.powi(2) + a(48) * x2 - a(36) * x1 * x2
                    + a(27) * x2.powi(2)))
}

/// Easom: Flat function with single sharp minimum.
///
/// Bounds: -50 <= xi <= 50 or other.
///
/// Optimum: f(pi, pi) = 0
///
/// ```
/// # #[macro_use] extern crate float_cmp;
/// # use ggtune::benchfn::easom;
/// # fn main() {
/// const PI32: f32 = std::f32::consts::PI;
/// const PI64: f64 = std::f64::consts::PI;
/// assert_eq!(easom(PI32, PI32, 1.0), 0.0, "optimum with f32");
/// assert_eq!(easom(PI64, PI64, 1.0), 0.0, "optimum with f64");
/// assert!(approx_eq!(f64, easom(0.0, 0.0, 1.0), 1.0, epsilon = 1e-5), "center is one");
/// assert!(approx_eq!(f64, easom(0.0, 0.0, 3.0), 3.0, epsilon = 1e-5), "can be scaled");
/// # }
/// ```
///
/// Definition taken from:
/// https://en.wikipedia.org/wiki/Test_functions_for_optimization
/// and adapted for non-negative outputs.
pub fn easom<A>(x1: A, x2: A, amplitude: A) -> A
where
    A: Float + FloatConst,
{
    amplitude
        * (A::one()
            - x1.cos() * x2.cos() * (-((x1 - A::PI()).powi(2) + (x2 - A::PI()).powi(2))).exp())
}

/// Himmelblau's function: Asymmetric polynomial with 4 minima.
///
/// Bounds: -5 <= xi <= 5
///
/// ```
/// # use ggtune::benchfn::himmelblau;
/// # #[macro_use] extern crate float_cmp;
/// # fn main() {
/// assert!(approx_eq!(f64, himmelblau(3.0, 2.0), 0.0, epsilon = 1E-5), "min 1");
/// assert!(approx_eq!(f64, himmelblau(-2.805118, 3.131312), 0.0, epsilon = 1E-5), "min 2");
/// assert!(approx_eq!(f64, himmelblau(-3.779310, -3.283186), 0.0, epsilon = 1E-5), "min 3");
/// assert!(approx_eq!(f64, himmelblau(3.584428, -1.848126), 0.0, epsilon = 1E-5), "min 4");
/// assert!(approx_eq!(f32, himmelblau(3.0, 2.0), 0.0, epsilon = 1E-5), "works with f32");
/// # }
/// ```
pub fn himmelblau<A>(x1: A, x2: A) -> A
where
    A: Float,
    u16: Into<A>,
{
    (x1.powi(2) + x2 - 11.into()).powi(2) + (x1 + x2.powi(2) - 7.into()).powi(2)
}

/// Rastrigin function: N-dimensional with many local minima.
///
/// Bounds: -5.12 <= xi <= 5.12
///
/// Optimum: f(0, ..., 0) = 0
///
/// ```
/// # #[macro_use] extern crate ndarray;
/// # use ggtune::benchfn::rastrigin;
/// # fn main() {
/// assert_eq!(rastrigin(array![0.0f32], 10.0), 0.0, "minimum in 1D as f32");
/// assert_eq!(rastrigin(array![0.0f64], 10.0), 0.0, "minimum in 1D as f64");
/// assert_eq!(rastrigin(array![0.0, 0.0], 10.0), 0.0, "minimum in 2D");
/// assert_eq!(rastrigin(vec![0.0; 10].into(), 10.0), 0.0, "minimum in 10D");
/// # }
/// ```
///
/// At least one dimension is required:
///
/// ``` should_panic
/// # #[macro_use] extern crate ndarray;
/// # use ggtune::benchfn::rastrigin;
/// # fn main() {
/// rastrigin(array![], 10.0);  // panics
/// # }
/// ```
pub fn rastrigin<A, S>(xs: ArrayBase<S, Ix1>, amplitude: A) -> A
where
    A: Float + FloatConst,
    S: Data<Elem = A>,
    i16: Into<A>,
{
    assert!(!xs.is_empty(), "at least one dimension required");
    assert!(amplitude > 0.into(), "amplitude must be strictly positive");

    let n_dim = xs.len() as i16;
    amplitude * n_dim.into()
        + xs.mapv(|x| x.powi(2) - amplitude * A::cos(2.into() * A::PI() * x))
            .sum()
}

/// Rosenbrock function: N-dimensional and asymmetric.
///
/// Bounds: unbounded, but interval [-2.5, 2.5] sensible.
///
/// Optimum: f(1, ..., 1) = 0
///
/// ```
/// # #[macro_use] extern crate ndarray;
/// # use ggtune::benchfn::rosenbrock;
/// # fn main() {
/// assert_eq!(rosenbrock(array![1.0, 1.0]), 0.0);
/// assert_eq!(rosenbrock(vec![1.0; 6].into()), 0.0);
/// # }
/// ```
///
/// At least two dimensions are required:
///
/// ``` should_panic
/// # #[macro_use] extern crate ndarray;
/// # use ggtune::benchfn::rosenbrock;
/// # fn main() {
/// rosenbrock(array![0.0]);  // panics
/// # }
/// ```
pub fn rosenbrock<A, S>(xs: ArrayBase<S, Ix1>) -> A
where
    A: Float,
    S: Data<Elem = A>,
    i16: Into<A>,
{
    assert!(xs.len() >= 2, "at least two dimensions required");

    fn slice<A, S, Slice>(xs: &ArrayBase<S, Ix1>, s: Slice) -> ArrayView1<A>
    where
        S: Data<Elem = A>,
        Slice: Into<ndarray::SliceOrIndex>,
    {
        xs.slice(&ndarray::SliceInfo::<_, Ix1>::new([s.into()]).unwrap())
    }

    let (xs, xnexts) = (slice(&xs, ..-1), slice(&xs, 1..));
    ((xnexts.to_owned() - xs.mapv(|x| x.powi(2)).mapv(|x| x.powi(2))).mapv(|x| 100.into() * x)
        + xs.mapv(|x| (1.into() - x).powi(2)))
    .sum()
}

/// One-Max function (just the sum of values).
///
/// Bounds: 0 <= xi <= 1
///
/// Optimum: f(0, ..., 0) = 0
///
/// ```
/// # use ggtune::benchfn::onemax;
/// assert_eq!(onemax(vec![0.0f32; 1].into()), 0.0, "optimum for f32 1D");
/// assert_eq!(onemax(vec![0.0f64; 1].into()), 0.0, "optimum for f64 1D");
/// assert_eq!(onemax(vec![0.0; 6].into()), 0.0, "optimum for 6D");
/// assert_eq!(onemax(vec![1.0; 1].into()), 1.0, "other point in 1D");
/// assert_eq!(onemax(vec![1.0; 6].into()), 6.0, "other point in 6D");
/// ```
pub fn onemax<S, A>(xs: ArrayBase<S, Ix1>) -> A
where
    A: Clone + std::ops::Add + num_traits::identities::Zero,
    S: Data<Elem = A>,
{
    xs.sum()
}
