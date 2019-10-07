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

mod logwarp {
    use crate::Scalar;
    use ndarray::prelude::*;
    use num_traits::Float;

    pub fn project_into<A>(y: Array1<A>) -> Array1<A>
    where
        A: Float,
    {
        y.mapv(Float::ln)
    }

    pub fn project_location_from<A>(y: Array1<A>) -> Array1<A>
    where
        A: Float,
    {
        y.mapv(Float::exp)
    }

    //// mean for lognormal distribution is `exp(mean + s^2/2)`
    pub fn project_mean_from<A>(mean: Array1<A>, variance: Array1<A>) -> Array1<A>
    where
        A: Scalar,
    {
        project_location_from(mean + variance / A::from_i(2))
    }

    #[test]
    fn test_project_mean_from() {
        let exp = f64::exp;

        let logmean = array![-1.0, 0.0, 0.5, 1.0, 0.0, 0.0];
        let logstd = array![1.0, 1.0, 1.0, 1.0, 0.5, 2.0];
        let actual = project_mean_from(logmean.clone(), logstd.mapv(|x| x * x));
        let expected = array![
            exp(-0.5),
            exp(0.5),
            exp(1.0),
            exp(1.5),
            exp(1.0 / 8.0),
            exp(2.0)
        ];
        const EPSILON: f64 = 1e-6;
        assert!(
            actual.all_close(&expected, EPSILON),
            "mean failed\n\
             in  logmean : {}\n\
             in  logstd  : {}\n\
             out actual  : {}\n\
             out expected: {}",
            logmean,
            logstd,
            actual,
            expected
        );
    }

    #[test]
    fn test_project_mean_std_from_data() {
        use float_cmp::ApproxEqRatio as _;
        use statrs::statistics::Statistics as _;
        let mut rng = crate::RNG::new_with_seed(83229);
        let data: Array1<f64> = std::iter::repeat_with(|| rng.normal(-1.0, 1.0).exp())
            .take(500)
            .collect();
        let logmean = data.mapv(Float::ln).mean();
        let logvar = data.mapv(Float::ln).population_variance();
        let actual_mean = *project_mean_from(array![logmean], array![logvar])
            .first()
            .unwrap();
        let actual_std = project_variance_from(array![logmean], array![logvar])
            .first()
            .unwrap()
            .sqrt();
        let expected_mean = data.mean();
        let expected_std = data.population_std_dev();
        // because we use noisy data, the comparison must be very rough
        assert!(
            expected_mean.approx_eq_ratio(&actual_mean, 0.05)
                & expected_std.approx_eq_ratio(&actual_std, 0.15),
            "test failed\n\
             mean   log: {:20}\n\
             | expected: {:20}\n\
             |   actual: {:20}\n\
             std    log: {:20}\n\
             | expected: {:20}\n\
             |   actual: {:20}",
            logmean,
            expected_mean,
            actual_mean,
            logvar.sqrt(),
            expected_std,
            actual_std,
        );
    }

    /// variance for lognormal distribution is `(exp(s^s) - 1) * exp(2 mu + s^2)`
    pub fn project_variance_from<A>(mean: Array1<A>, variance: Array1<A>) -> Array1<A>
    where
        A: Scalar,
    {
        (mean * A::from_i(2) + &variance).mapv(Float::exp)
            * (variance.mapv(Float::exp) - A::from_i(1))
    }

    #[test]
    fn test_project_variance() {
        let exp = f64::exp;
        let sqrt = f64::sqrt;

        let logmean = array![-1.0, 0.0, 0.5, 1.0, 0.0, 0.0];
        let logstd = array![1.0, 1.0, 1.0, 1.0, 0.5, 2.0];
        let actual = project_variance_from(logmean.clone(), logstd.mapv(|x| x * x)).mapv(sqrt);
        let expected = project_mean_from(logmean.clone(), logstd.mapv(|x| x * x))
            * logstd.mapv(|x| sqrt(exp(x * x) - 1.0));
        const EPSILON: f64 = 1e-7;
        assert!(
            actual.all_close(&expected, EPSILON),
            "variance failed (shown as std):\n\
             in  logmean : {}\n\
             in  logstd  : {}\n\
             out actual  : {}\n\
             out expected: {}",
            logmean,
            logstd,
            actual,
            expected
        );
    }

    /// cv for lognormal distribution is `sqrt(exp(s^2) - 1)`
    pub fn project_cv_from<A>(variance: Array1<A>) -> Array1<A>
    where
        A: Scalar,
    {
        (variance.mapv(Float::exp) - A::from_i(1)).mapv(Float::sqrt)
    }
}

impl<A> YNormalize<A>
where
    A: crate::Scalar,
{
    pub fn new_project_into_normalized(y: Array1<A>, projection: Projection) -> (Array1<A>, Self) {
        match projection {
            Projection::Linear => {
                let expected = guess_min(y.view(), A::from_i(0));
                let y = y - expected;
                let amplitude = guess_amplitude(y.view());
                let y = y / amplitude + A::from_f(FUDGE_MIN);

                let cfg = YNormalize {
                    amplitude,
                    expected,
                    projection,
                };
                (y, cfg)
            }
            Projection::Logarithmic => {
                let expected = guess_min(y.view(), A::from_f(1.0));
                let y = logwarp::project_into(y - expected);
                let amplitude = guess_amplitude(y.view());
                let y = y / amplitude;

                let cfg = YNormalize {
                    amplitude,
                    expected,
                    projection,
                };
                (y, cfg)
            }
        }
    }

    pub fn project_into_normalized(&self, y: Array1<A>) -> Array1<A>
    where
        A: crate::Scalar,
    {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            Projection::Linear => (y - expected) / amplitude + A::from_f(FUDGE_MIN),
            Projection::Logarithmic => logwarp::project_into(y - expected) / amplitude,
        }
    }

    /// Project a value back from the normalized range to the natural range.
    /// Using this function is OK for observations or quantiles.
    /// It is not OK for mean, variance, etc.
    pub fn project_location_from_normalized(&self, y: Array1<A>) -> Array1<A> {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            Projection::Linear => (y - A::from_f(FUDGE_MIN)) * amplitude + expected,
            Projection::Logarithmic => logwarp::project_location_from(y * amplitude) + expected,
        }
    }

    pub fn project_mean_from_normalized(
        &self,
        mean: Array1<A>,
        variance: ArrayView1<A>,
    ) -> Array1<A> {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            Projection::Linear => (mean - A::from_f(FUDGE_MIN)) * amplitude + expected,
            // Treat this as the mean of a log-normal distribution:
            // mean is exp(mu + (sigma^2)/2)
            Projection::Logarithmic => {
                let mean_amp = mean * amplitude;
                let var_amp = variance.to_owned() * amplitude * amplitude;
                logwarp::project_mean_from(mean_amp, var_amp) + expected
            }
        }
    }

    pub fn project_std_from_normalized(
        &self,
        mean: ArrayView1<A>,
        variance: Array1<A>,
    ) -> Array1<A> {
        let Self {
            amplitude,
            expected: _,
            projection,
        } = *self;
        match projection {
            Projection::Linear => variance.mapv(Float::sqrt) * amplitude,
            Projection::Logarithmic => {
                let mu = mean.to_owned() * amplitude;
                let sigma2 = variance * amplitude * amplitude;
                logwarp::project_variance_from(mu, sigma2).mapv(Float::sqrt)
            }
        }
    }

    pub fn project_cv_from_normalized(
        &self,
        mean: ArrayView1<A>,
        variance: Array1<A>,
    ) -> Array1<A> {
        let Self {
            amplitude,
            expected,
            projection,
        } = *self;
        match projection {
            // cv for normal distribution is sigma/mu
            Projection::Linear => {
                variance.mapv(Float::sqrt) * amplitude
                    / ((mean.to_owned() - A::from_f(FUDGE_MIN)) * amplitude + expected)
            }
            Projection::Logarithmic => logwarp::project_cv_from(variance * amplitude.powi(2)),
        }
    }
}

/// select a value so that all ys can be made non-negative
fn guess_min<A>(y: ArrayView1<A>, minimum: A) -> A
where
    A: crate::Scalar,
{
    *y.min().unwrap() - minimum
}

/// select a value so that the mean can be normalized to 1, as long as any deviation exists
fn guess_amplitude<A>(y: ArrayView1<A>) -> A
where
    A: crate::Scalar,
{
    let amplitude = y.mean_axis(Axis(0)).into_scalar();
    if amplitude > A::from_f(0.0) {
        amplitude
    } else {
        A::from_f(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_inverse(input: impl Into<Array1<f64>>, projection: Projection) {
        let input = input.into();
        let (y, norm) = YNormalize::new_project_into_normalized(input.clone(), projection);
        let rey = norm.project_into_normalized(input.clone());
        let result = norm.project_location_from_normalized(y.clone());
        assert!(
            input.all_close(&result, 1e-4),
            "result should be close to input\n\
             input : {}\n\
             result: {}",
            input,
            result
        );
        assert!(
            y.all_close(&rey, 1e-4),
            "project_into variants should produce same result\n\
             new_project_into_normalized : {}\n\
             norm.project_into_normalized: {}",
            y,
            rey
        );
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
        let std = norm.project_std_from_normalized(Array1::zeros(3).view(), array![0.0, 1.0, 4.0]);
        let expected = array![0.0, 3.0, 6.0];
        assert!(
            std.all_close(&expected, 1e-4),
            "expected: {} but got: {}",
            expected,
            std
        );
    }

    fn test_logarithmic_project_mean(
        norm: &YNormalize<f64>,
        projected_mean: f64,
        projected_std: f64,
        expected_mean: f64,
    ) {
        let actual_mean = *norm
            .project_mean_from_normalized(
                array![projected_mean],
                array![projected_std.powi(2)].view(),
            )
            .first()
            .unwrap();
        assert!(
            approx_eq!(f64, actual_mean, expected_mean, epsilon = 1E-7),
            "expected {} but got {}\n\
             for test norm: {:?}, mean: {} std: {}",
            expected_mean,
            actual_mean,
            norm,
            projected_mean,
            projected_std
        );
    }

    #[test]
    fn logarithmic_project_mean_into_normalized_standard() {
        let norm = YNormalize {
            expected: 0.0,
            amplitude: 1.0,
            projection: Projection::Logarithmic,
        };
        test_logarithmic_project_mean(&norm, 0.5, 1.0, std::f64::consts::E);
        test_logarithmic_project_mean(&norm, 2.0, 3.0, f64::exp(2.0 + 9.0 / 2.0));
    }

    #[test]
    fn logarithmic_project_mean_into_normalized_scaled() {
        let norm = YNormalize {
            expected: 0.0,
            amplitude: 2.0,
            projection: Projection::Logarithmic,
        };
        test_logarithmic_project_mean(&norm, 0.5, 1.0, std::f64::consts::E.powi(3));
        test_logarithmic_project_mean(&norm, 1.0, 1.5, f64::exp(2.0 + 9.0 / 2.0));
    }

    #[test]
    fn logarithmic_project_mean_into_normalized_translated() {
        let norm = YNormalize {
            expected: 1.0,
            amplitude: 1.0,
            projection: Projection::Logarithmic,
        };
        test_logarithmic_project_mean(&norm, 0.5, 1.0, std::f64::consts::E + 1.0);
    }

    fn test_statistics(projection: Projection, mean_ratio: f64, std_ratio: f64) {
        use float_cmp::ApproxEqRatio as _;
        use statrs::statistics::Statistics as _;

        // let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 2.0];
        let mut rng = crate::core::random::RNG::new_with_seed(903282318);
        let data: Array1<f64> = std::iter::repeat_with(|| rng.normal(3.0, 1.0).exp())
            .take(500)
            .collect();
        let expected_mean = data.mean();
        let expected_std = data.population_std_dev();
        let expected_cv = expected_std / expected_mean;

        let (transformed, norm) = YNormalize::new_project_into_normalized(data.clone(), projection);

        let transformed_mean = transformed.mean();
        let transformed_variance = transformed.population_variance();

        let actual_mean = *norm
            .project_mean_from_normalized(
                array![transformed_mean],
                array![transformed_variance].view(),
            )
            .first()
            .unwrap();
        let actual_std = *norm
            .project_std_from_normalized(
                array![transformed_mean].view(),
                array![transformed_variance],
            )
            .first()
            .unwrap();
        let actual_cv = *norm
            .project_cv_from_normalized(
                array![transformed_mean].view(),
                array![transformed_variance],
            )
            .first()
            .unwrap();

        let cv_ratio = ((mean_ratio).powi(2) + (std_ratio).powi(2)).sqrt();
        assert!(
            actual_mean.approx_eq_ratio(&expected_mean, mean_ratio)
                & actual_std.approx_eq_ratio(&expected_std, std_ratio)
                & actual_cv.approx_eq_ratio(&expected_cv, cv_ratio),
            "projections failed to produce expected results\n\
             mean got: {:20} expected: {:20}\n\
             std  got: {:20} expected: {:20}\n\
             cv   got: {:20} expected: {:20} (ratio {})",
            actual_mean,
            expected_mean,
            actual_std,
            expected_std,
            actual_cv,
            expected_cv,
            cv_ratio,
        );
    }

    #[test]
    fn linear_test_statistics() {
        test_statistics(Projection::Linear, 1E-7, 1E-7);
    }

    #[test]
    fn logarithmic_test_statistics() {
        test_statistics(Projection::Logarithmic, 0.01, 0.1);
    }
}

impl std::str::FromStr for Projection {
    type Err = failure::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Ok(match input {
            "lin" | "linear" => Projection::Linear,
            "log" | "ln" | "logarithmic" => Projection::Logarithmic,
            _ => {
                return Err(format_err!(
                    "unknown name for Projection, must be {{lin, linear, log, ln, logarithmic}}"
                ))
            }
        })
    }
}
