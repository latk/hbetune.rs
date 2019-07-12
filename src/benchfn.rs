use ndarray::prelude::*;
use ndarray::Data;
use num_traits::Float;

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
where S: Data<Elem = A>,
      A: Clone + Float,
{
    assert!(xs.len() > 0, "at least one dimension required");
    xs.mapv(|x| x.powi(2)).sum()
}
