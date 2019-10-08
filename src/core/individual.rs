use crate::ParameterValue;
use itertools::Itertools as _;
use serde::{Deserialize, Serialize};

type Generation = i64;

/// Parameters and result of a pending or completed experiment.
/// Many fields are write-once (checked dynamically).
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Individual<A> {
    gen: Option<Generation>,
    observation: Option<A>,
    prediction: Option<A>,
    #[serde(rename = "ei")]
    expected_improvement: Option<A>,
    cost: Option<A>,
    sample: Vec<ParameterValue>,
}

impl<A> Individual<A> {
    /// Create a new individual at a certain sample.
    pub fn new(sample: Vec<ParameterValue>) -> Self {
        Individual {
            sample,
            gen: None,
            prediction: None,
            expected_improvement: None,
            observation: None,
            cost: None,
        }
    }

    /// Check whether all write-once fields have been provided.
    /// If true, the object is effectively immutable.
    pub fn is_fully_initialized(&self) -> bool {
        self.gen.is_some()
            && self.prediction.is_some()
            && self.expected_improvement.is_some()
            && self.observation.is_some()
            && self.cost.is_some()
    }

    /// The input variables at which the experiment was evaluated
    /// or shall be evaluated.
    pub fn sample(&self) -> &[ParameterValue] {
        self.sample.as_slice()
    }

    /// The generation in which the Individual was evaluated. Write-once.
    pub fn gen(&self) -> Option<Generation> {
        self.gen
    }

    /// The predicted value (write once).
    pub fn prediction(&self) -> Option<A>
    where
        A: Copy,
    {
        self.prediction
    }

    /// The expected improvement before the Individual was evaluated. Write-once.
    pub fn expected_improvement(&self) -> Option<A>
    where
        A: Copy,
    {
        self.expected_improvement
    }

    /// Write the predicted value and the expected improvement.
    pub fn set_prediction_and_ei(&mut self, prediction: A, ei: A) -> Result<(), ()> {
        fail_if_some(self.prediction.replace(prediction))?;
        fail_if_some(self.expected_improvement.replace(ei))
    }

    /// The observed value (write once).
    pub fn observation(&self) -> Option<A>
    where
        A: Copy,
    {
        self.observation
    }

    /// The observed cost (write once).
    pub fn cost(&self) -> Option<A>
    where
        A: Copy,
    {
        self.cost
    }

    /// Write the evaluation result: generation, observation, and cost.
    pub fn set_evaluation_result(
        &mut self,
        gen: Generation,
        observation: A,
        cost: A,
    ) -> Result<(), ()> {
        fail_if_some(self.gen.replace(gen))?;
        fail_if_some(self.observation.replace(observation))?;
        fail_if_some(self.cost.replace(cost))
    }
}

fn fail_if_some<T>(maybe: Option<T>) -> Result<(), ()> {
    match maybe {
        Some(_) => Err(()),
        None => Ok(()),
    }
}

/// Format an Individual for debugging.
///
/// ```
/// # #[macro_use] extern crate ndarray;
/// # extern crate ggtune;
/// # use ndarray::prelude::*;
/// # use ggtune::Individual;
/// let mut ind: Individual<f64> = Individual::new(
///     vec![1.0.into(), 2.0.into(), 3.0.into()],
/// );
/// assert_eq!(
///     format!("{:?}", ind),
///     "Individual(None @None [Real(1.0) Real(2.0) Real(3.0)] prediction: None ei: None gen: None)",
/// );
/// ind.set_prediction_and_ei(1.2, 0.3);
/// ind.set_evaluation_result(3, 17.2, 0.2);
/// assert_eq!(
///     format!("{:?}", ind),
///     "Individual(17.2 @0.20 [Real(1.0) Real(2.0) Real(3.0)] prediction: 1.2 ei: 0.3 gen: 3)",
/// );
/// ```
impl<A> std::fmt::Debug for Individual<A>
where
    A: std::fmt::Display + std::fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        struct FlatOption<T>(Option<T>);

        impl<T> std::fmt::Display for FlatOption<T>
        where
            T: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match &self.0 {
                    Some(x) => x.fmt(f),
                    None => f.write_str("None"),
                }
            }
        }

        write!(
            f,
            "Individual({observation} @{cost:.2} [{sample:?}] \
             prediction: {prediction} \
             ei: {expected_improvement} \
             gen: {gen})",
            observation = FlatOption(self.observation),
            cost = FlatOption(self.cost),
            sample = self.sample.iter().format(" "),
            prediction = FlatOption(self.prediction),
            expected_improvement = FlatOption(self.expected_improvement),
            gen = FlatOption(self.gen),
        )
    }
}
