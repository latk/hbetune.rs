//! A Gaussian Process Guided Tuner

#[macro_use]
extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;
extern crate num_traits;
extern crate openblas_src;
#[cfg(test)] extern crate speculate;
#[macro_use] extern crate itertools;

#[macro_use]mod macros;
mod util;

mod gpr;
mod hierarchical;
mod kernel;
mod knn;
mod minimize;
mod outputs;
mod random;
mod space;
mod surrogate_model;

pub mod benchfn;

pub use surrogate_model::SurrogateModel;
pub use gpr::{SurrogateModelGPR, ConfigGPR};
pub use space::Space;
pub use random::RNG;
