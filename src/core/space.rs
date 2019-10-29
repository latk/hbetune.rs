use itertools::Itertools as _;
use ndarray::prelude::*;
use rand::seq::SliceRandom as _;
use serde::{Deserialize, Serialize};

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

    /// Add a log-scale real-valued parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// The parameter `offset` defaults to zero, and must be chosen so that `lo + offset > 0`.
    /// Panics if the parameter range has zero size or is otherwise invalid.
    pub fn add_logreal_parameter(
        &mut self,
        name: impl Into<String>,
        lo: f64,
        hi: f64,
        offset: Option<f64>,
    ) {
        let offset = offset.unwrap_or(0.0);
        assert!(lo < hi, "logreal parameter: range must not be empty");
        assert!(
            lo + offset > 0.0,
            "logreal parameter: lo + offset must be > 0"
        );
        self.add_parameter(Parameter::LogReal {
            name: name.into(),
            lo,
            hi,
            offset,
        });
    }

    /// Add a log-scaled integer parameter.
    /// The bounds `lo`, `hi` are inclusive.
    /// The parameter `offset` defaults to zero, and must be chosen so that `lo + offset > 0`.
    /// Panics if the parameter range is empty or is otherwise invalid.
    pub fn add_logint_parameter(
        &mut self,
        name: impl Into<String>,
        lo: i64,
        hi: i64,
        offset: Option<i64>,
    ) {
        let offset = offset.unwrap_or(0);
        assert!(lo < hi, "logint parameter: range must not be empty");
        assert!(
            lo + offset > 0,
            "logreal parameter: lo + offset must be > 0"
        );
        self.add_parameter(Parameter::LogInt {
            name: name.into(),
            lo,
            hi,
            offset,
        })
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
            .map(Array1::from)
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
            .map(Individual::new)
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
        (0..n)
            .map(|i| choices.iter().map(|samples| samples[i]).collect())
            .collect()
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

fn project_into_range(x: f64, lo: f64, hi: f64) -> f64 {
    // clip the transformed value to prevent numerical precision problems
    crate::util::clip((x - lo) / (hi - lo), Some(0.0), Some(1.0))
}

fn project_from_range(x: f64, lo: f64, hi: f64) -> f64 {
    // clip the transformed value to prevent numerical precision problems
    crate::util::clip(x * (hi - lo) + lo, Some(lo), Some(hi))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    Int(i64),
    Real(f64),
}

impl ParameterValue {
    pub fn to_f64(self) -> f64 {
        match self {
            ParameterValue::Real(x) => x,
            ParameterValue::Int(x) => x as f64,
        }
    }
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
        self.to_f64()
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
    Real {
        name: String,
        lo: f64,
        hi: f64,
    },
    Int {
        name: String,
        lo: i64,
        hi: i64,
    },
    LogReal {
        name: String,
        lo: f64,
        hi: f64,
        offset: f64,
    },
    LogInt {
        name: String,
        lo: i64,
        hi: i64,
        offset: i64,
    },
}

impl Parameter {
    pub fn name(&self) -> &str {
        match self {
            Parameter::Real { name, .. } => name,
            Parameter::Int { name, .. } => name,
            Parameter::LogReal { name, .. } => name,
            Parameter::LogInt { name, .. } => name,
        }
    }

    fn project_into_features<A: Scalar>(&self, x: ParameterValue) -> A {
        match *self {
            Parameter::Real { lo, hi, ref name } => match x {
                ParameterValue::Real(x) => A::from_f(project_into_range(x, lo, hi)),
                x => unreachable!("Real({}): cannot project {:?} into features", name, x),
            },
            Parameter::Int { lo, hi, ref name } => match x {
                // Why not "... / (hi - lo + 1)"?
                // So that the min/max integers match the min/max feature space bounds.
                // The corresponding rounding areas still match.
                ParameterValue::Int(x) => A::from_f(((x - lo) as f64) / (hi - lo) as f64),
                x => unreachable!("Int({}): cannot project {:?} into features", name, x),
            },
            Parameter::LogReal {
                lo,
                hi,
                offset,
                ref name,
            } => match x {
                ParameterValue::Real(x) => A::from_f(project_into_range(
                    (x + offset).ln(),
                    (lo + offset).ln(),
                    (hi + offset).ln(),
                )),
                x => unreachable!("LogReal({}): cannot project {:?} into features", name, x),
            },
            Parameter::LogInt {
                lo,
                hi,
                offset,
                ref name,
            } => match x {
                ParameterValue::Int(x) => {
                    let logx = ((x + offset) as f64).ln();
                    let loglo = ((lo + offset) as f64).ln();
                    let logsize = ((hi + offset) as f64).ln() - loglo;
                    let feature = (logx - loglo) / logsize;
                    A::from_f(crate::util::clip(feature, Some(0.0), Some(1.0)))
                }
                x => unreachable!("LogInt({}): cannot project {:?} into features", name, x),
            },
        }
    }

    fn project_from_features<A: Scalar>(&self, x: A) -> ParameterValue {
        match *self {
            Parameter::Real { lo, hi, .. } => {
                ParameterValue::Real(project_from_range(x.into(), lo, hi))
            }
            Parameter::Int { lo, hi, .. } => ParameterValue::Int(crate::util::clip(
                (x.into() * (hi - lo + 1) as f64).floor() as i64 + lo,
                Some(lo),
                Some(hi),
            )),
            Parameter::LogReal { lo, hi, offset, .. } => ParameterValue::Real(
                project_from_range(x.into(), (lo + offset).ln(), (hi + offset).ln()).exp() - offset,
            ),
            Parameter::LogInt { lo, hi, offset, .. } => {
                let loglo = ((lo + offset) as f64).ln();
                let logsize = ((hi + offset + 1) as f64).ln() - loglo;
                let x = (x.into() * logsize + loglo).exp().floor() as i64 - offset;
                ParameterValue::Int(crate::util::clip(x, Some(lo), Some(hi)))
            }
        }
    }

    fn mutate_inplace(&self, x: &mut ParameterValue, relscale: f64, rng: &mut RNG) {
        match *self {
            Parameter::Real { lo, hi, ref name } => match *x {
                ParameterValue::Real(ref mut x) => {
                    *x = sample_truncnorm(*x, relscale * (hi - lo), lo, hi, rng)
                }
                x => unreachable!("Real({}): cannot mutate {:?}", name, x),
            },
            Parameter::Int { .. } | Parameter::LogInt { .. } => match x {
                x @ ParameterValue::Int { .. } => {
                    *x = self.project_from_features(sample_truncnorm(
                        self.project_into_features(*x),
                        relscale,
                        0.0,
                        1.0,
                        rng,
                    ))
                }
                x => unreachable!("{:?}: cannot mutate {:?}", *self, *x),
            },
            Parameter::LogReal { .. } => match x {
                x @ ParameterValue::Real { .. } => {
                    *x = self.project_from_features(sample_truncnorm(
                        self.project_into_features(*x),
                        relscale,
                        0.0,
                        1.0,
                        rng,
                    ))
                }
                x => unreachable!("{:?}: cannot mutate {:?}", *self, *x),
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
            Parameter::Real { .. }
            | Parameter::Int { .. }
            | Parameter::LogReal { .. }
            | Parameter::LogInt { .. } => sample_n_in_unit_range(n, rng)
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

macro_rules! with_error_context {
    ($err:ident => ($format:literal $(, $arg:expr)*) $body:block) => {
        (|| -> ::std::result::Result<_, failure::Error> { Ok($body) })()
            .with_context(|$err| format!($format $(, $arg)*))
    }
}

impl std::str::FromStr for Parameter {
    type Err = failure::Error;

    fn from_str(s: &str) -> Result<Parameter, Self::Err> {
        use failure::ResultExt as _;

        fn no_more_items(mut items: impl Iterator) -> Result<(), failure::Error> {
            if items.next().is_some() {
                Err(format_err!("too many items"))
            } else {
                Ok(())
            }
        }

        fn parse_next<T, Iter>(
            symbol: &str,
            items: &mut Iter,
        ) -> Result<T, failure::Context<String>>
        where
            Iter: Iterator,
            Iter::Item: AsRef<str>,
            T: std::str::FromStr,
            <T as std::str::FromStr>::Err: failure::Fail,
        {
            with_error_context!(err => ("while parsing '{}': {}", symbol, err) {
                items
                    .next()
                    .ok_or_else(|| format_err!("value is missing"))?
                    .as_ref()
                    .parse()?
            })
        }

        fn parse_next_option<T, Iter>(
            symbol: &str,
            items: &mut Iter,
        ) -> Result<Option<T>, failure::Context<String>>
        where
            Iter: Iterator,
            Iter::Item: AsRef<str>,
            T: std::str::FromStr,
            <T as std::str::FromStr>::Err: failure::Fail,
        {
            with_error_context!(err => ("while parsing '{}': {}", symbol, err) {
                items
                    .next()
                    .map(|s| -> Result<T, _> { s.as_ref().parse() })
                    .transpose()?
            })
        }

        // split on whitespace and colons
        let mut items = s
            .split_whitespace()
            .flat_map(|tok| tok.split(':'))
            .filter(|s| !s.is_empty());
        let (name, the_type) = items
            .next()
            .and_then(|name| items.next().map(|the_type| (name.to_owned(), the_type)))
            .ok_or_else(|| {
                format_err!(
                    "while parsing '<name> <type> <...>': value is missing (input: {:?})",
                    s
                )
            })?;

        let result = match the_type {
            "real" => with_error_context!(
            err => ("while parsing 'real <lo> <hi>': {}", err) {
                let lo = parse_next("<lo>", &mut items)?;
                let hi = parse_next("<hi>", &mut items)?;
                no_more_items(items)?;
                Parameter::Real { name, lo, hi }
            })
            .map_err(Into::into),

            "int" => with_error_context!(
            err => ("while parsing 'int <lo> <hi>': {}", err) {
                let lo = parse_next("<lo>", &mut items)?;
                let hi = parse_next("<hi>", &mut items)?;
                no_more_items(items)?;
                Parameter::Int { name, lo, hi }
            })
            .map_err(Into::into),

            "logreal" => with_error_context!(
            err => ("while parsing 'logreal <lo> <hi> [<offset>]': {}", err) {
                let lo = parse_next("<lo>", &mut items)?;
                let hi = parse_next("<hi>", &mut items)?;
                let offset = parse_next_option("<offset>", &mut items)?.unwrap_or(0.0);
                no_more_items(items)?;
                Parameter::LogReal{ name, lo, hi, offset }
            })
            .map_err(Into::into),

            "logint" => with_error_context!(
            err => ("while parsing 'logint <lo> <hi> [<offset>]: {}", err) {
                let lo = parse_next("<lo>", &mut items)?;
                let hi = parse_next("<hi>", &mut items)?;
                let offset = parse_next_option("<offset>", &mut items)?.unwrap_or(0);
                no_more_items(items)?;
                Parameter::LogInt { name, lo, hi, offset }
            })
            .map_err(Into::into),

            t => Err(format_err!(
                "type must be one of real/int/logreal/logint, was: {}",
                t
            )),
        };
        Ok(result.with_context(|err| format!("{} (input: {:?})", err, s))?)
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
        let feature_mid: f64 = param.project_into_features(ParameterValue::Int(2));

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

        assert!(
            approx_eq!(f64, feature_min, 0.0),
            "feature should reach min bound"
        );
        assert!(
            approx_eq!(f64, feature_max, 1.0),
            "feature should reach max bound"
        );

        assert!(
            approx_eq!(f64, feature_mid, 0.5),
            "midpoint value should be midpoint of feature space"
        );

        for x in -2..=6 {
            let value = ParameterValue::Int(x);
            let feature = param.project_into_features(value);
            let roundtrip = param.project_from_features(feature);
            assert_eq!(roundtrip, value, "must roundtrip for x={}", x);
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
                x => unreachable!("only Int() should be possible: {:?}", x),
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

    #[test]
    fn project_logreal() {
        let param = Parameter::LogReal {
            lo: -1.0,
            hi: 10.0,
            offset: 1.5,
            name: "foo".to_owned(),
        };

        let feature_min = param.project_into_features(ParameterValue::Real(-1.0));
        let feature_max = param.project_into_features(ParameterValue::Real(10.0));
        assert!(
            approx_eq!(f64, feature_min, 0.0),
            "feature should reach min bound",
        );
        assert!(
            approx_eq!(f64, feature_max, 1.0),
            "feature should reach max bound",
        );

        for x in -1..=10 {
            let value = ParameterValue::Real(f64::from(x));
            let feature = param.project_into_features(value);
            let roundtrip = param.project_from_features(feature);
            assert!(
                approx_eq!(f64, roundtrip.into(), value.into()),
                "must roundtrip for x={}",
                x,
            );
            assert!(
                0.0 <= feature && feature <= 1.0,
                "feature {} must stay within bounds [0, 1]",
                feature,
            );
        }
    }

    #[test]
    fn project_logint() {
        let param = Parameter::LogInt {
            lo: -1,
            hi: 10,
            offset: 2,
            name: "foo".to_owned(),
        };

        let feature_min = param.project_into_features(ParameterValue::Int(-1));
        let feature_max = param.project_into_features(ParameterValue::Int(10));
        assert!(
            approx_eq!(f64, feature_min, 0.0),
            "feature should reach min bound",
        );
        assert!(
            approx_eq!(f64, feature_max, 1.0),
            "feature should reach max bound",
        );

        for x in -1..=10 {
            let value = ParameterValue::Int(x);
            let feature = param.project_into_features(value);
            let roundtrip = param.project_from_features(feature);
            assert!(
                approx_eq!(f64, roundtrip.into(), value.into()),
                "must roundtrip for x={}",
                x,
            );
            assert!(
                0.0 <= feature && feature <= 1.0,
                "feature {} must stay within bounds [0, 1]",
                feature,
            );
        }
    }
}
