use itertools::Itertools as _;
use ndarray::prelude::*;
use rand::seq::SliceRandom as _;

use crate::{Individual, Scalar, RNG};

/// The parameter design space that contains feasible solutions.
/// This space is usually handled in its natural units.
/// Internally, samples are projected into a design space of reals in the range 0 to 1 inclusive.
#[derive(Debug, Clone, Default)]
pub struct Space {
    params: Vec<Parameter>,
}

impl Space {
    /// Create a new parameter space.
    pub fn new() -> Self {
        Space::default()
    }

    /// The number of dimensions.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    // The space is empty if no parameters were provided.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// The parameters, in order.
    pub fn params(&self) -> &[Parameter] {
        self.params.as_slice()
    }

    /// Add a real-valued parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// Panics if the parameter range has zero size.
    pub fn add_real_parameter(&mut self, name: impl Into<String>, lo: f64, hi: f64) {
        assert!(lo < hi);
        self.add_parameter(Parameter::Real {
            name: name.into(),
            lo,
            hi,
        });
    }

    /// Add an integer-valued parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// Panics if the parameter range has zero size.
    pub fn add_integer_parameter(&mut self, name: impl Into<String>, lo: i64, hi: i64) {
        assert!(lo < hi);
        self.add_parameter(Parameter::Int {
            name: name.into(),
            lo,
            hi,
        });
    }

    pub fn add_parameter(&mut self, param: Parameter) {
        self.params.push(param);
    }

    /// Project a parameter values into the feature space.
    ///
    /// ```
    /// #[macro_use] extern crate ndarray;
    /// # use ggtune::Space;
    /// # fn main() {
    /// let mut space = Space::new();
    /// space.add_real_parameter("a", -2.0, 2.0);
    /// space.add_real_parameter("b", 0.0, 10.0);
    /// let params = [(-1.0).into(), 7.0.into()];
    /// let features: Vec<f64> = space.project_into_features(params);
    /// assert_eq!(features, &[0.25, 0.7]);
    /// # }
    /// ```
    pub fn project_into_features<A: Scalar>(&self, x: impl AsRef<[ParameterValue]>) -> Vec<A> {
        let x = x.as_ref();
        assert!(
            x.len() == self.params.len(),
            "the space has {} parameters but got {} values",
            self.params.len(),
            x.len(),
        );
        x.iter()
            .zip_eq(&self.params)
            .map(|(x, param)| param.project_into_features(*x))
            .collect()
    }

    pub fn project_into_features_array<A: Scalar, Sample: AsRef<[ParameterValue]>>(
        &self,
        xs: impl IntoIterator<Item = Sample>,
    ) -> Array2<A> {
        let features = xs
            .into_iter()
            .map(|x| self.project_into_features(x.as_ref()))
            .map(|x| Array1::from(x))
            .collect_vec();
        ndarray::stack(
            Axis(0),
            features
                .iter()
                .map(|x| x.view().insert_axis(Axis(0)))
                .collect_vec()
                .as_slice(),
        )
        .unwrap()
    }

    /// Get a parameter vector back from the feature space.
    ///
    /// ```
    /// #[macro_use] extern crate ndarray;
    /// # use ggtune::{Space, ParameterValue};
    /// # fn main() {
    /// let mut space = Space::new();
    /// space.add_real_parameter("a", -2.0, 2.0);
    /// space.add_real_parameter("b", 0.0, 10.0);
    /// let params = [(-1.0).into(), 7.0.into()];
    /// let features: Vec<f64> = space.project_into_features(params.as_ref());
    /// assert_eq!(
    ///     space.project_from_features(features).as_slice(),
    ///     params.as_ref(),
    /// );
    /// # }
    /// ```
    pub fn project_from_features<A: Scalar>(&self, x: impl AsRef<[A]>) -> Vec<ParameterValue> {
        x.as_ref()
            .iter()
            .zip_eq(&self.params)
            .map(|(x, param)| param.project_from_features(*x))
            .collect()
    }

    /// Sample a new Individual uniformly from the projected parameter space.
    pub fn sample_individual<A>(&self, rng: &mut RNG) -> Individual<A>
    where
        A: Scalar + rand::distributions::uniform::SampleUniform,
    {
        Individual::new(self.sample(rng))
    }

    pub fn sample_individual_n<A>(&self, n: usize, rng: &mut RNG) -> Vec<Individual<A>> {
        self.sample_n(n, rng)
            .into_iter()
            .map(|sample| Individual::new(sample))
            .collect()
    }

    pub fn sample(&self, rng: &mut RNG) -> Vec<ParameterValue> {
        self.sample_n(1, rng).into_iter().next().unwrap()
    }

    /// Obtain multiple evenly-distributed random samples.
    ///
    /// Uses Latin Hypercube Sampling to cover the entire parameter space with even probability.
    pub fn sample_n(&self, n: usize, rng: &mut RNG) -> Vec<Vec<ParameterValue>> {
        let choices = self
            .params
            .iter()
            .map(|param| {
                let mut samples = param.sample_n(n, rng);
                assert_eq!(samples.len(), n, "sample must contain n elements");
                samples.shuffle(rng.basic_rng_mut());
                samples
            })
            .collect_vec();
        let samples = (0..n)
            .map(|i| choices.iter().map(|samples| samples[i]).collect())
            .collect();
        samples
    }

    pub fn mutate_inplace(&self, sample: &mut [ParameterValue], relscale: &[f64], rng: &mut RNG) {
        assert_eq!(sample.len(), relscale.len());
        assert_eq!(sample.len(), self.len());

        for (i, x) in sample.iter_mut().enumerate() {
            self.params[i].mutate_inplace(x, relscale[i], rng);
        }
    }
}

fn sample_truncnorm(mu: f64, sigma: f64, a: f64, b: f64, rng: &mut RNG) -> f64 {
    use statrs::distribution::{InverseCDF, Normal, Univariate};

    let normal = Normal::new(0.0, 1.0).unwrap();
    let az = (a - mu) / sigma;
    let bz = (b - mu) / sigma;

    let xz = rng.uniform(normal.cdf(az)..=normal.cdf(bz));

    normal.inverse_cdf(xz) * sigma + mu
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterValue {
    Real(f64),
    Int(i64),
}

impl From<f64> for ParameterValue {
    fn from(x: f64) -> ParameterValue {
        ParameterValue::Real(x)
    }
}

impl From<i64> for ParameterValue {
    fn from(x: i64) -> ParameterValue {
        ParameterValue::Int(x)
    }
}

impl Into<f64> for ParameterValue {
    fn into(self) -> f64 {
        match self {
            ParameterValue::Real(x) => x,
            ParameterValue::Int(x) => x as f64,
        }
    }
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ParameterValue::Real(x) => write!(fmt, "{}", x),
            ParameterValue::Int(x) => write!(fmt, "{}", x),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Parameter {
    Real { name: String, lo: f64, hi: f64 },
    Int { name: String, lo: i64, hi: i64 },
}

impl Parameter {
    pub fn name(&self) -> &str {
        match self {
            Parameter::Real { name, .. } => name,
            Parameter::Int { name, .. } => name,
        }
    }

    fn project_into_features<A: Scalar>(&self, x: ParameterValue) -> A {
        match *self {
            Parameter::Real { lo, hi, ref name } => match x {
                ParameterValue::Real(x) => A::from_f((x - lo) / (hi - lo)),
                x @ _ => unreachable!("Real({}): cannot project {:?} into features", name, x),
            },
            Parameter::Int { lo, hi, ref name } => match x {
                // Why not "... / (hi - lo + 1)"?
                // So that the min/max integers match the min/max feature space bounds.
                // The corresponding rounding areas still match.
                ParameterValue::Int(x) => A::from_f(((x - lo) as f64) / (hi - lo) as f64),
                x @ _ => unreachable!("Int({}): cannot project {:?} into features", name, x),
            },
        }
    }

    fn project_from_features<A: Scalar>(&self, x: A) -> ParameterValue {
        match *self {
            Parameter::Real { lo, hi, .. } => ParameterValue::Real(x.into() * (hi - lo) + lo),
            Parameter::Int { lo, hi, .. } => ParameterValue::Int(crate::util::clip(
                (x.into() * (hi - lo + 1) as f64).floor() as i64 + lo,
                Some(lo),
                Some(hi),
            )),
        }
    }

    fn mutate_inplace(&self, x: &mut ParameterValue, relscale: f64, rng: &mut RNG) {
        match *self {
            Parameter::Real { lo, hi, ref name } => match *x {
                ParameterValue::Real(ref mut x) => *x = sample_truncnorm(*x, relscale, lo, hi, rng),
                x @ _ => unreachable!("Real({}): cannot mutate {:?}", name, x),
            },
            Parameter::Int { ref name, .. } => match x {
                x @ ParameterValue::Int { .. } => {
                    *x = self.project_from_features(sample_truncnorm(
                        self.project_into_features(*x),
                        relscale,
                        0.0,
                        1.0,
                        rng,
                    ))
                }
                x @ _ => unreachable!("Int({}): cannot mutate {:?}", name, *x),
            },
        }
    }

    /// Obtain random samples of that parameter.
    ///
    /// If multiple samples are requested,
    /// each sample is taken from an equi-distributed band
    /// so that the full parameter space is evenly covered.
    fn sample_n(&self, n: usize, rng: &mut RNG) -> Vec<ParameterValue> {
        let out = match *self {
            Parameter::Real { lo, hi, .. } => {
                assert!(lo < hi, "lo {} must be lower than hi {}", lo, hi);
                sample_n_in_unit_range(n, rng)
                    .into_iter()
                    .map(|x| self.project_from_features(x))
                    .collect_vec()
            }
            Parameter::Int { .. } => sample_n_in_unit_range(n, rng)
                .into_iter()
                .map(|x| self.project_from_features(x))
                .collect_vec(),
        };
        assert_eq!(out.len(), n, "must have requested size");
        out
    }
}

fn sample_n_in_unit_range(n: usize, rng: &mut RNG) -> Vec<f64> {
    let bounds = (0..n).map(|x| x as f64 / n as f64).collect_vec();

    let last_window = bounds.last().cloned().unwrap_or(0.0)..=1.0;
    assert!(
        last_window.start() < last_window.end(),
        "window for last sample must not be empty: {:?}",
        last_window,
    );
    // select a sample in each window
    let last_item = std::iter::once(rng.uniform(last_window)).take((n > 0) as usize);
    let n_minus_one_items = bounds[..bounds.len()]
        .iter()
        .zip(&bounds[1..])
        .map(|(&window_lo, &window_hi)| rng.uniform(window_lo..window_hi));

    n_minus_one_items.chain(last_item).collect_vec()
}

impl std::str::FromStr for Parameter {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Parameter, Self::Err> {
        use failure::ResultExt as _;
        let err_too_few_items = || {
            format_err!(
                "too few items, expected: '<name> <type> <...>' but got: {}",
                s
            )
        };
        let err_real_too_few_items = || {
            format_err!(
                "too few items, expected: '<name> real <lo> <hi>' but got: {}",
                s
            )
        };
        let err_real_too_many_items = || {
            format_err!(
                "too many items, expected: '<name> real <lo> <hi>' but got: {}",
                s
            )
        };
        let err_int_too_few_items = || {
            format_err!(
                "too few items, expected: '<name> int <lo> <hi>' but got: {}",
                s,
            )
        };
        let err_int_too_many_items = || {
            format_err!(
                "too many items, expected: '<name> int <lo> <hi>' but got: {}",
                s,
            )
        };

        let mut items = s.split_whitespace();
        let name: String = items.next().ok_or_else(err_too_few_items)?.to_owned();
        let the_type: &str = items.next().ok_or_else(err_too_few_items)?;
        match the_type {
            "real" => {
                let lo = items
                    .next()
                    .ok_or_else(err_real_too_few_items)?
                    .parse::<f64>()
                    .with_context(|err| format!("while parsing <lo>: {}", err))?;

                let hi = items
                    .next()
                    .ok_or_else(err_real_too_few_items)?
                    .parse::<f64>()
                    .with_context(|err| format!("while parsing <hi>: {}", err))?;

                if items.next().is_some() {
                    return Err(err_real_too_many_items());
                }

                Ok(Parameter::Real { name, lo, hi })
            }
            "int" => {
                let lo = items
                    .next()
                    .ok_or_else(err_int_too_few_items)?
                    .parse::<i64>()
                    .with_context(|err| format!("while parsing <lo>: {}", err))?;

                let hi = items
                    .next()
                    .ok_or_else(err_int_too_few_items)?
                    .parse::<i64>()
                    .with_context(|err| format!("while parsing <hi>: {}", err))?;

                if items.next().is_some() {
                    return Err(err_int_too_many_items());
                }

                Ok(Parameter::Int { name, lo, hi })
            }
            t => bail!("type must be 'real', was: {}", t),
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Parameter, ParameterValue, RNG};

    #[test]
    fn project_int() {
        let param = Parameter::Int {
            lo: -2,
            hi: 6,
            name: "foo".to_owned(),
        };

        let feature_min: f64 = param.project_into_features(ParameterValue::Int(-2));
        let feature_max: f64 = param.project_into_features(ParameterValue::Int(6));

        assert_eq!(
            param.project_from_features(0.0),
            ParameterValue::Int(-2),
            "must project lower bound from features",
        );

        assert_eq!(
            param.project_from_features(1.0),
            ParameterValue::Int(6),
            "must project upper bound from features",
        );

        assert_eq!(feature_min, 0.0, "feature should reach min bound");
        assert_eq!(feature_max, 1.0, "feature should reach max bound");

        assert_eq!(
            param.project_into_features::<f64>(ParameterValue::Int(2)),
            0.5,
            "midpoint value should be midpoint of feature space",
        );

        for x in -2..=6 {
            let value = ParameterValue::Int(x);
            let feature = param.project_into_features(value);
            assert_eq!(
                param.project_from_features(feature),
                value,
                "must roundtrip for x={}",
                x
            );
            assert!(
                feature_min <= feature,
                "feature {} satisfies lower bound {} (x={})",
                feature,
                feature_min,
                x
            );
            assert!(
                feature <= feature_max,
                "feature {} satisfies upper bound {} (x={})",
                feature,
                feature_max,
                x
            );
        }
    }

    #[test]
    fn sample_int() {
        let param = Parameter::Int {
            lo: 1,
            hi: 3,
            name: "whatever".to_owned(),
        };
        let mut rng = RNG::new_with_seed(378);
        let mut counts = [0; 3];
        const EXPECTED_SAMPLES: usize = 20;
        for value in param.sample_n(3 * EXPECTED_SAMPLES, &mut rng) {
            match value {
                ParameterValue::Int(x) => {
                    counts[(x - 1) as usize] += 1;
                }
                x @ _ => unreachable!("only Int() should be possible: {:?}", x),
            }
        }
        for &count in &counts {
            assert!(
                (EXPECTED_SAMPLES - 3) <= count && count <= (EXPECTED_SAMPLES + 3),
                "counts must be roughly equal: {:?}",
                counts
            );
        }
    }
}
