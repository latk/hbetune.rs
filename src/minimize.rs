use crate::acquisition::{AcquisitionStrategy, MutationAcquisition};
use crate::gpr::{SurrogateModelGPR, EstimatorGPR};
use crate::individual::Individual;
use crate::kernel::Scalar;
use crate::outputs::{OutputEventHandler, Output};
use crate::random::RNG;
use crate::space::Space;
use crate::surrogate_model::{SurrogateModel, Estimator as ModelEstimator};
use itertools::Itertools as _;
use ndarray::prelude::*;
use std::iter::FromIterator as _;
use std::marker::PhantomData;
use std::time::{Instant, Duration};

type TimeSource = Fn() -> Instant;

type ObjectiveFunction<A> = Fn(ArrayView1<A>, &mut RNG) -> (f64, f64);

pub struct OptimizationResult<A, Model> {
    all_individuals: Box<Vec<Individual<A>>>,
    all_models: Vec<Model>,
    duration: Duration,
}

impl<A, Model> OptimizationResult<A, Model> {
    fn new(
        all_individuals: impl IntoIterator<Item = Individual<A>>,
        all_models: impl IntoIterator<Item = Model>,
        duration: Duration,
    ) -> Self {
        let all_individuals = Box::new(all_individuals.into_iter().collect_vec());
        let all_models = Vec::from_iter(all_models);
        Self { all_individuals, all_models, duration }
    }

    /// All results.
    pub fn all_individuals(&self) -> &[Individual<A>] {
        self.all_individuals.as_ref()
    }

    /// All models.
    pub fn all_models(&self) -> &[Model] {
        self.all_models.as_ref()
    }

    /// Total duration
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Select the best evaluation results.
    pub fn best_n(&self, n: usize) -> impl Iterator<Item = &Individual<A>>
    where A: PartialOrd {
        self.all_individuals.iter()
            .filter(|ind| ind.observation.is_some())
            .sorted_by(|a, b| a.observation.partial_cmp(&b.observation)
                       .expect("observation should have an order"))
            .take(n)
    }

    /// Best result.
    pub fn best_individual(&self) -> Option<&Individual<A>>
    where A: PartialOrd {
        self.best_n(1).next()
    }

    /// Final model.
    pub fn model(&self) -> Option<&Model> {
        self.all_models.last()
    }

    /// Input variables of all evaluations
    pub fn xs(&self) -> Array2<A>
    where A: Copy {
        let samples = self.all_individuals.iter()
            .map(|ind| ind.sample.view().insert_axis(Axis(0)))
            .collect_vec();
        ndarray::stack(Axis(0), &samples).unwrap()
    }

    /// Output variables of all evaluations.
    pub fn ys(&self) -> Option<Array1<A>>
    where A: Clone {
        let ys: Option<Vec<A>> = self.all_individuals.iter()
            .map(|ind| ind.observation.clone())
            .collect();
        Some(Array::from_vec(ys?))
    }

    /// Best observed value.
    pub fn fmin(&self) -> Option<A>
    where A: Clone + PartialOrd {
        self.best_individual().and_then(|ind| ind.observation.clone())
    }
}

/// Configuration for the minimizer.
pub struct Minimizer<A, Estimator>
where Estimator: ModelEstimator<A>,
      A: Scalar,
{
    /// How many samples are taken per generation.
    pub popsize: usize,

    /// How many initial samples should be acquired
    /// before model-guided acquisition takes over.
    pub initial: usize,

    /// How many samples may be taken in total per optimization run.
    pub max_nevals: usize,

    /// Standard deviation for creating new samples,
    /// as fraction of each parameter's range.
    pub relscale_initial: f64,

    /// Factor by which the relscale is reduced per generation.
    pub relscale_attenuation: f64,

    /// An estimator for the regression model to fit the response surface.
    pub estimator: Option<Estimator>,

    /// How new samples are acquired.
    pub acquisition_strategy: Option<Box<dyn AcquisitionStrategy<A>>>,

    /// Whether the model prediction should be used as a fitness function
    /// when selecting which samples proceed to the next generation.
    /// If false, the objective's observed value incl. noise is used.
    pub select_via_posterior: bool,

    /// Whether the model prediction is used
    /// to find the current best point during optimization.
    /// If false, the objective's observed value incl. noise is used.
    pub fmin_via_posterior: bool,

    /// How many random samples are suggested per generation.
    /// Usually, new samples are created by random mutations of existing samples.
    pub n_replacements: usize,

    pub time_source: Box<TimeSource>,
}

impl<A, Estimator> Default for Minimizer<A, Estimator>
where Estimator: ModelEstimator<A>,
      A: Scalar,
{
    fn default() -> Self {
        Self {
            popsize: 10,
            initial: 10,
            max_nevals: 100,
            relscale_initial: 0.3,
            relscale_attenuation: 0.9,
            estimator: None,
            acquisition_strategy: None,
            time_source: Box::new(|| Instant::now()),
            select_via_posterior: false,
            fmin_via_posterior: true,
            n_replacements: 1,
        }
    }
}

impl<A, Estimator> Minimizer<A, Estimator>
where Estimator: ModelEstimator<A>,
      A: Scalar,
{
    /// Minimize the objective `objective(sample, rng) -> (value, cost)`.
    ///
    /// outputs: controls what information is printed during optimization.
    /// Can e.g. be used to save evaluations in a CSV file.
    ///
    /// historic_individuals: previous evaluations of the same objective/space
    /// that should be incorporated into the model.
    /// Can be useful in order to benefit from previous minimization runs.
    /// Potential drawbacks include a biased model,
    /// and that the tuner slows down with additional samples.
    /// Constraints: all individuals must be fully initialized,
    /// and declare the -1th generation.
    pub fn minimize(
        mut self,
        objective: Box<ObjectiveFunction<A>>,
        space: Space,
        rng: &mut RNG,
        outputs: Option<Box<dyn OutputEventHandler<A>>>,
        historic_individuals: impl IntoIterator<Item = Individual<A>>,
    ) -> OptimizationResult<A, Estimator::Model>
    {
        assert!(self.initial + self.popsize <= self.max_nevals,
                "evaluation budget {max_nevals} too small with {initial}+n*{popsize} evaluations",
                max_nevals = self.max_nevals, initial = self.initial, popsize = self.popsize);

        self.acquisition_strategy.get_or_insert_with(
            || Box::new(MutationAcquisition { breadth: 10 })
        );

        let outputs = outputs.unwrap_or_else(
            || Box::new(Output::new(&space))
        );

        let historic_individuals = Vec::from_iter(historic_individuals);

        assert!(historic_individuals.iter()
                .all(|ind| ind.is_fully_initialized()));
        assert!(historic_individuals.iter()
                .all(|ind| ind.gen.expect("gen should be initialized") == -1));

        let instance = MinimizationInstance {
            config: self, objective, space, outputs,
        };

        instance.run(rng, historic_individuals)
    }
}

struct MinimizationInstance<A, Estimator>
where Estimator: ModelEstimator<A>,
      A: Scalar,
{
    config: Minimizer<A, Estimator>,
    objective: Box<ObjectiveFunction<A>>,
    space: Space,
    outputs: Box<dyn OutputEventHandler<A>>,
}

impl<A, Estimator> MinimizationInstance<A, Estimator>
where Estimator: ModelEstimator<A>,
      A: Scalar,
{
    fn run(&self, rng: &mut RNG, historic_individuals: Vec<Individual<A>>) -> OptimizationResult<A, Estimator::Model> {
        unimplemented!("MinimizationInstance::run()")
    }
}
