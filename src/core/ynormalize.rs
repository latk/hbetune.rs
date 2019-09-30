use ndarray::prelude::*;
use ndarray_stats::QuantileExt as _;
use num_traits::Float;

const FUDGE_MIN: f64 = 0.05;

#[derive(Debug, Clone, PartialEq)]
pub struct YNormalize<A> {
    amplitude: A,
    expected: A,
    projection: Projection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Logarithmic,
    Linear,
}

impl<A> YNormalize<A>
where
    A: crate::Scalar,
{
    pub fn project_into_normalized(y: Array1<A>, projection: Projection) -> (Array1<A>, Self) {
        match projection {
            Projection::Linear => {
                let (expected, y) = normalize_min(y, None);
                let (amplitude, y) = normalize_mean(y);
                let y = y + A::from_f(FUDGE_MIN);

                let cfg = YNormalize {
                    amplitude,
                    expected,
                    projection,
                };
                (y, cfg)
            }
            Projection::Logarithmic => {
                let (expected, y) = normalize_min(y, Some(A::from_f(1.0)));
                let y = y.mapv(Float::ln);
                let (amplitude, y) = normalize_mean(y);

                let cfg = YNormalize {
                    amplitude,
                    expected,
                    projection,
                };
                (y, cfg)
            }
        }
    }

    pub fn project_from_normalized(&self, y: Array1<A>) -> Array1<A> {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            Projection::Linear => {
                let y = y - A::from_f(FUDGE_MIN);
                let y = denormalize_mean(y, amplitude);
                let y = denormalize_min(y, expected);
                y
            }
            Projection::Logarithmic => {
                let y = denormalize_mean(y, amplitude);
                let y = y.mapv(Float::exp);
                let y = denormalize_min(y, expected);
                y
            }
        }
    }

    pub fn project_std_from_normalized_variance(&self, variance: Array1<A>) -> Array1<A> {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            Projection::Linear => variance.mapv(Float::sqrt) * amplitude,
            Projection::Logarithmic => {
                // note: this cannot be implemented because the scale is not known.
                // Instead, the code would have to be changed to report confidence bounds.
                unimplemented!("project std from normalized variance, logarithmic")
            }
        }
    }
}

/// shift the values so that all are non-negative
fn normalize_min<A>(y: Array1<A>, minimum: Option<A>) -> (A, Array1<A>)
where
    A: crate::Scalar,
{
    let expected = *y.min().unwrap() - minimum.unwrap_or(A::from_f(0.0));
    (expected, y - expected)
}

fn denormalize_min<A>(y: Array1<A>, expected: A) -> Array1<A>
where
    A: crate::Scalar,
{
    y + expected
}

/// scale the values so that the mean is set to 1, as long as there is any deviation
fn normalize_mean<A>(y: Array1<A>) -> (A, Array1<A>)
where
    A: crate::Scalar,
{
    let amplitude = y.mean_axis(Axis(0)).into_scalar();

    if amplitude > A::from_f(0.0) {
        (amplitude, y / amplitude)
    } else {
        (A::from_f(1.0), y)
    }
}

fn denormalize_mean<A>(y: Array1<A>, amplitude: A) -> Array1<A>
where
    A: crate::Scalar,
{
    y * amplitude
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_inverse(input: impl Into<Array1<f64>>, projection: Projection) {
        let input = input.into();
        let (y, norm) = YNormalize::project_into_normalized(input.clone(), projection);
        let result = norm.project_from_normalized(y);
        assert!(input.all_close(&result, 1e-4));
    }

    #[test]
    fn linear_has_an_inverse_operation() {
        test_inverse(vec![1., 2., 3., 4.], Projection::Linear);
    }

    #[test]
    fn logarithmic_has_an_inverse_operation() {
        test_inverse(vec![1., 2., 3., 4.], Projection::Logarithmic);
    }

    #[test]
    fn linear_also_works_with_negative_numbers() {
        test_inverse(vec![-5., 3., 8., -2.], Projection::Linear);
    }

    #[test]
    fn logarithmic_also_works_with_negative_numbers() {
        test_inverse(vec![-5., 3., 8., -2.], Projection::Linear);
    }

    #[test]
    fn linear_can_handle_variance() {
        let norm = YNormalize {
            expected: 1234.0,
            amplitude: 3.0,
            projection: Projection::Linear,
        };
        let std = norm.project_std_from_normalized_variance(array![0.0, 1.0, 4.0]);
        let expected = array![0.0, 3.0, 6.0];
        assert!(
            std.all_close(&expected, 1e-4),
            "expected: {} but got: {}",
            expected,
            std
        );
    }
}
