//! Kernel code adapted from the sklearn.gaussian_process.kernels Python module.

mod constant_kernel;
mod fit;
mod kernel;
mod lml;
mod matern_kernel;
mod predict;
mod product_kernel;
mod scalar;

pub use constant_kernel::ConstantKernel;
pub use fit::FittedKernel;
pub use kernel::Kernel;
pub use lml::LmlWithGradient;
pub use matern_kernel::Matern;
pub use predict::predict;
pub use product_kernel::Product;
pub use scalar::Scalar;
