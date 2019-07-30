//! A Gaussian Process Guided Tuner

#[macro_use] extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;
extern crate num_traits;
extern crate openblas_src;
extern crate itertools;
extern crate rayon;
extern crate statrs;

#[cfg(test)] extern crate speculate;

#[macro_use]mod macros;
mod util;

mod acquisition;
mod gpr;
mod hierarchical;
mod individual;
mod kernel;
mod knn;
mod minimize;
mod outputs;
mod random;
mod space;
mod surrogate_model;

pub mod benchfn;

pub use kernel::Scalar;
pub use surrogate_model::{SurrogateModel, Estimator};
pub use gpr::{SurrogateModelGPR, EstimatorGPR};
pub use space::Space;
pub use random::RNG;
pub use individual::Individual;
pub use minimize::{Minimizer, OptimizationResult, ObjectiveFunction};
pub use outputs::{OutputEventHandler, CompositeOutputEventHandler, Output};
pub use acquisition::{AcquisitionStrategy, MutationAcquisition};
