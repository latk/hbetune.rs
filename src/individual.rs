use itertools::Itertools as _;
use ndarray::prelude::*;

type Generation = i64;

/// Parameters and result of a pending or completed experiment.
/// Many fields are write-once (checked dynamically).
#[derive(Clone)]
pub struct Individual<A> {
    sample: Array1<A>,
    gen: Option<Generation>,
    prediction: Option<A>,
    expected_improvement: Option<A>,
    observation: Option<A>,
    cost: Option<A>,
}

impl<A> Individual<A>
where
    A: Copy,
{
    /// Create a new individual at a certain sample.
    pub fn new(sample: impl Into<Array1<A>>) -> Self {
        Individual {
            sample: sample.into(),
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
    pub fn sample(&self) -> ArrayView1<A> {
        self.sample.view()
    }

    /// The generation in which the Individual was evaluated. Write-once.
    pub fn gen(&self) -> Option<Generation> {
        self.gen
    }

    /// Write the generation in which the Individual was evaluated.
    pub fn set_gen(&mut self, gen: Generation) -> Result<(), Generation> {
        fail_if_some(self.gen.replace(gen))
    }

    /// The predicted value (write once).
    pub fn prediction(&self) -> Option<A> {
        self.prediction
    }

    /// Write the predicted value.
    pub fn set_prediction(&mut self, prediction: A) -> Result<(), A> {
        fail_if_some(self.prediction.replace(prediction))
    }

    /// The expected improvement before the Individual was evaluated. Write-once.
    pub fn expected_improvement(&self) -> Option<A> {
        self.expected_improvement
    }

    /// Write the expected improvement.
    pub fn set_expected_improvement(&mut self, ei: A) -> Result<(), A> {
        fail_if_some(self.expected_improvement.replace(ei))
    }

    /// The observed value (write once).
    pub fn observation(&self) -> Option<A> {
        self.observation.clone()
    }

    /// Write the observed value.
    pub fn set_observation(&mut self, observation: A) -> Result<(), A> {
        fail_if_some(self.observation.replace(observation))
    }

    /// The observed cost (write once).
    pub fn cost(&self) -> Option<A> {
        self.cost.clone()
    }

    /// Write the observed cost.
    pub fn set_cost(&mut self, cost: A) -> Result<(), A> {
        fail_if_some(self.cost.replace(cost))
    }
}

fn fail_if_some<T>(maybe: Option<T>) -> Result<(), T> {
    match maybe {
        Some(x) => Err(x),
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
/// let mut ind: Individual<f64> = Individual::new(array![1.0, 2.0, 3.0]);
/// ind.set_observation(17.2);
/// ind.set_expected_improvement(0.3);
/// assert_eq!(format!("{:?}", ind),
///            "Individual(17.2 @None [1.0 2.0 3.0] prediction: None ei: 0.3 gen: None)");
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
