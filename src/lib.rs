//! A Gaussian Process Guided Tuner

#[macro_use]
extern crate approx;
extern crate csv;
#[macro_use]
extern crate failure;
extern crate itertools;
extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;
extern crate num_traits;
extern crate openblas_src;
extern crate rayon;
extern crate serde;
extern crate statrs;
#[macro_use]
extern crate structopt;

#[macro_use]
mod util;

mod core;
mod gpr;

pub use crate::core::acquisition::{
    AcquisitionStrategy, MutationAcquisition, SearchingAcquisition,
};
pub use crate::core::benchfn;
pub use crate::core::gpr::{EstimatorGPR, SurrogateModelGPR};
pub use crate::core::individual::Individual;
pub use crate::core::minimize::{
    Minimizer, MinimizerArgs, ObjectiveFunction, ObjectiveFunctionFromFn, OptimizationResult,
};
pub use crate::core::outputs::{DurationCounter, Output, OutputEventHandler};
pub use crate::core::random::RNG;
pub use crate::core::space::{Parameter, ParameterValue, Space};
pub use crate::core::surrogate_model::{Estimator, SurrogateModel};
pub use crate::core::ynormalize::Projection;
pub use crate::gpr::Scalar;
