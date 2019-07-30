//! A Gaussian Process Guided Tuner

#[macro_use]
extern crate float_cmp;
#[macro_use]
extern crate ndarray;
extern crate itertools;
extern crate ndarray_stats;
extern crate noisy_float;
extern crate num_traits;
extern crate openblas_src;
extern crate rayon;
extern crate statrs;

#[cfg(test)]
extern crate speculate;

#[macro_use]
mod macros;
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

pub use acquisition::{AcquisitionStrategy, MutationAcquisition};
pub use gpr::{EstimatorGPR, SurrogateModelGPR};
pub use individual::Individual;
pub use kernel::Scalar;
pub use minimize::{Minimizer, ObjectiveFunction, OptimizationResult};
pub use outputs::{CompositeOutputEventHandler, Output, OutputEventHandler};
pub use random::RNG;
pub use space::Space;
pub use surrogate_model::{Estimator, SurrogateModel};
