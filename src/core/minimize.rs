use crate::core::surrogate_model::SummaryStatistics;
use crate::{AcquisitionStrategy, MutationAcquisition, SearchingAcquisition};
use crate::{Estimator as ModelEstimator, SurrogateModel};
use crate::{Individual, Output, OutputEventHandler as _, ParameterValue, Scalar, Space, RNG};

use std::iter::FromIterator as _;
use std::time::{Duration, Instant};

use itertools::Itertools as _;
use ndarray::prelude::*;

type TimeSource = dyn Fn() -> Instant;

pub trait ObjectiveFunction<A>: Sync {
    fn run<'a>(&self, xs: &[ParameterValue], rng: &'a mut RNG) -> (A, A);
}

pub struct ObjectiveFunctionFromFn<F>(F);

impl<F> ObjectiveFunctionFromFn<F> {
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<A, F> ObjectiveFunction<A> for ObjectiveFunctionFromFn<F>
where
    F: Fn(&[ParameterValue]) -> A + Sync,
    A: Default,
{
    fn run<'a>(&self, xs: &[ParameterValue], _rng: &'a mut RNG) -> (A, A) {
        let Self(f) = self;
        (f(xs), Default::default())
    }
}

pub struct OptimizationResult<A, Model> {
    all_individuals: Vec<Individual<A>>,
    all_models: Vec<Model>,
    duration: Duration,
    suggestion_location: Vec<ParameterValue>,
    suggestion_statistics: SummaryStatistics<A>,
}

impl<A, Model> OptimizationResult<A, Model> {
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

    /// Location about the suggested location of the minimum.
    pub fn suggestion_location(&self) -> &[ParameterValue] {
        self.suggestion_location.as_slice()
    }

    /// Information about the suggested location of the minimum.
    pub fn suggestion_statistics(&self) -> &SummaryStatistics<A> {
        &self.suggestion_statistics
    }

    /// Final model.
    pub fn model(&self) -> Option<&Model> {
        self.all_models.last()
    }

    /// Output variables of all evaluations.
    pub fn ys(&self) -> Option<Array1<A>>
    where
        A: Copy,
    {
        let ys: Option<Vec<A>> = self
            .all_individuals
            .iter()
            .map(|ind| ind.observation())
            .collect();
        Some(Array::from(ys?))
    }

    /// Best observed value.
    pub fn fmin(&self) -> Option<A>
    where
        A: Copy + PartialOrd,
    {
        self.all_individuals()
            .iter()
            .map(|ind| ind.observation().expect("should have observation"))
            .min_by(|a, b| a.partial_cmp(b).expect("observation should have an order"))
    }
}

/// Configuration for the minimizer.
#[derive(StructOpt, Debug)]
#[structopt(rename_all = "kebab")]
pub struct Minimizer {
    /// How many samples are taken per generation.
    #[structopt(long, default_value = "10")]
    pub popsize: usize,

    /// How many initial samples should be acquired
    /// before model-guided acquisition takes over.
    #[structopt(long, default_value = "10")]
    pub initial: usize,

    /// How many samples are taken for validation of the optimum.
    #[structopt(long, default_value = "1")]
    pub validation: usize,

    /// How many samples may be taken in total per optimization run.
    #[structopt(long, default_value = "100")]
    pub max_nevals: usize,

    /// Standard deviation for creating new samples,
    /// as fraction of each parameter's range.
    #[structopt(long, default_value = "0.3")]
    pub relscale_initial: f64,

    /// Factor by which the relscale is reduced per generation.
    #[structopt(long, default_value = "0.9")]
    pub relscale_attenuation: f64,

    /// The fitness function used to select which samples proceed to the next generation.
    /// If posterior/prediction/model: use trained model.
    /// If observation: use observed value incl. noise.
    #[structopt(long, default_value = "observation")]
    pub select_via: FitnessVia,

    /// How to select the minimum fitness (fmin).
    /// If posterior/prediction/model: use trained model.
    /// If observation: use observed value incl. noise.
    #[structopt(long, default_value = "prediction")]
    pub fmin_via: FitnessVia,

    /// Mean rate at which rejected individuals are given a second chance
    /// when selecting between parents and offspring for a new population.
    /// Towards zero, there is only offspring-parent competition.
    /// Towards one, all individuals compete against each other.
    #[structopt(long, default_value = "0.25")]
    pub competition_rate: f64,

    /// Penalize uncertainty in the objective prediction when suggesting the best sample.
    /// Only used during final suggestion phase, not during iterative optimization.
    /// Specifies the "number of sigmas" of the upper confidence bound,
    /// e.g. `1` is the 1-sigma confidence bound on the objective.
    /// Setting to zero corresponds to optimizing the *median* predicted objective.
    #[structopt(long, default_value = "1.0")]
    pub confidence_bound: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum FitnessVia {
    Prediction,
    Observation,
}

impl std::str::FromStr for FitnessVia {
    type Err = failure::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "posterior" | "prediction" | "model" => Ok(FitnessVia::Prediction),
            "observation" => Ok(FitnessVia::Observation),
            _ => bail!(
                "expected posterior/prediction/model or observation, but got: {:?}",
                s
            ),
        }
    }
}

impl Default for Minimizer {
    fn default() -> Self {
        Self {
            popsize: 10,
            initial: 10,
            validation: 1,
            max_nevals: 100,
            relscale_initial: 0.3,
            relscale_attenuation: 0.9,
            select_via: FitnessVia::Observation,
            fmin_via: FitnessVia::Prediction,
            competition_rate: 0.25,
            confidence_bound: 1.0,
        }
    }
}

pub struct MinimizerArgs<'life, A, Estimator> {
    pub estimator: Option<Estimator>,

    pub acquisition_strategy: Option<Box<dyn AcquisitionStrategy<A>>>,

    /// Controls what information is printed during optimization.
    /// Can e.g. be used to save evaluations in a CSV file.
    pub output: Output<'life, A>,

    /// Previous evaluations of the same objective/space
    /// that should be incorporated into the model.
    /// Can be useful in order to benefit from previous minimization runs.
    /// Potential drawbacks include a biased model,
    /// and that the tuner slows down with additional samples.
    /// Constraints: all individuals must be fully initialized,
    /// and declare the -1th generation.
    pub historic_individuals: Vec<Individual<A>>,

    pub time_source: Option<Box<TimeSource>>,
}

impl<A, Estimator> Default for MinimizerArgs<'_, A, Estimator> {
    fn default() -> Self {
        Self {
            estimator: None,
            acquisition_strategy: None,
            output: Output::new(),
            historic_individuals: vec![],
            time_source: None,
        }
    }
}

impl Minimizer {
    /// Minimize the objective `objective(sample, rng) -> (value, cost)`.
    pub fn minimize<'life, A, Estimator>(
        self,
        objective: &'life dyn ObjectiveFunction<A>,
        space: Space,
        rng: &'life mut RNG,
        args: MinimizerArgs<'life, A, Estimator>,
    ) -> Result<OptimizationResult<A, Estimator::Model>, Estimator::Error>
    where
        A: Scalar,
        Estimator: ModelEstimator<A>,
    {
        let MinimizerArgs {
            estimator,
            acquisition_strategy,
            mut output,
            historic_individuals,
            time_source,
        } = args;

        assert!(
            self.initial + self.validation <= self.max_nevals
                && self.popsize <= self.max_nevals,
            "evaluation budget {max_nevals} too small with {initial}+n*{popsize}+{validation} evaluations",
            max_nevals = self.max_nevals,
            initial = self.initial,
            popsize = self.popsize,
            validation = self.validation,
        );

        assert!(
            0.0 <= self.competition_rate && self.competition_rate <= 1.0,
            "competition rate {} must be in range [0, 1]",
            self.competition_rate,
        );

        let acquisition_strategy = acquisition_strategy.unwrap_or_else(|| {
            Box::new(SearchingAcquisition {
                base: MutationAcquisition { breadth: 10 },
            })
        });

        let time_source = time_source.unwrap_or_else(|| Box::new(Instant::now));

        let estimator = estimator.unwrap_or_else(|| Estimator::new(&space));

        let historic_individuals = Vec::from_iter(historic_individuals);

        assert!(historic_individuals
            .iter()
            .all(|ind| ind.is_fully_initialized()));
        assert!(historic_individuals
            .iter()
            .all(|ind| ind.gen().expect("gen should be initialized") == -1));

        let instance = MinimizationInstance {
            config: self,
            objective,
            space,
            acquisition_strategy,
            time_source,
            estimator,
        };

        instance.run(rng, &mut output, historic_individuals)
    }
}

struct MinimizationInstance<'life, A, Estimator>
where
    Estimator: ModelEstimator<A>,
    A: Scalar,
{
    config: Minimizer,
    objective: &'life dyn ObjectiveFunction<A>,
    space: Space,
    acquisition_strategy: Box<dyn AcquisitionStrategy<A> + 'life>,
    time_source: Box<TimeSource>,
    estimator: Estimator,
}

impl<'life, A, Estimator> MinimizationInstance<'life, A, Estimator>
where
    Estimator: ModelEstimator<A>,
    A: Scalar,
{
    fn run(
        self,
        rng: &mut RNG,
        output: &mut Output<'life, A>,
        historic_individuals: Vec<Individual<A>>,
    ) -> Result<OptimizationResult<A, Estimator::Model>, Estimator::Error> {
        let config = &self.config;

        let total_duration = self.time_source.as_ref()();

        let mut population = self.make_initial_population(rng);

        let mut budget = config.max_nevals.saturating_sub(config.validation);

        let mut all_evalutions: Vec<Individual<A>> = Vec::new();
        let mut all_models: Vec<Estimator::Model> = Vec::new();

        all_evalutions.extend(historic_individuals);

        for ind in &mut population {
            ind.set_prediction_and_ei(A::from_f(0.0), A::from_f(0.0))
                .expect("individual cannot have previous prediction");
        }

        self.evaluate_all(population.as_mut_slice(), rng, output, 0);
        budget = budget.saturating_sub(population.len());
        all_evalutions.extend(population.iter().cloned());

        all_models.push(self.fit_next_model(all_evalutions.as_slice(), 0, None, rng, output)?);

        let find_fmin = |individuals: &[Individual<A>], model: &Estimator::Model| {
            let fmin_operator = FitnessOperator(model, &self.space, config.fmin_via);
            individuals
                .iter()
                .min_by(|a, b| {
                    fmin_operator
                        .compare(a, b)
                        .expect("individuals should be orderable")
                })
                .and_then(|ind| fmin_operator.get_fitness(ind))
        };

        let mut fmin: A = find_fmin(all_evalutions.as_slice(), all_models.last().unwrap())
            .expect("fmin could be found");

        // Counter for the current generation.
        // The u16 type is large enough for expected values (0 .. a few hundred)
        // and small enough to be convertible to i32, usize
        let mut generation: u16 = 0;

        while budget > 0 {
            generation += 1;
            let model = all_models.last().unwrap();
            population = self.resize_population(
                population,
                std::cmp::min(budget, config.popsize),
                &model,
                rng,
            );

            use crate::util::clip;
            let relscale_bound = self.relscale_at_gen(generation.into());
            let relscale = model
                .length_scales()
                .iter()
                .map(|scale| clip(*scale, None, Some(relscale_bound)))
                .collect_vec();

            output.event_new_generation(generation.into(), relscale.as_slice());

            let mut offspring = self.acquire(
                population.as_slice(),
                &model,
                rng,
                output,
                fmin,
                relscale.as_slice(),
            );
            assert_eq!(population.len(), offspring.len());

            self.evaluate_all(offspring.as_mut_slice(), rng, output, generation);
            budget = budget.saturating_sub(offspring.len());
            all_evalutions.extend(offspring.iter().cloned());

            let next_model = self.fit_next_model(
                all_evalutions.as_slice(),
                generation.into(),
                Some(model),
                rng,
                output,
            )?;
            all_models.push(next_model);
            let model = all_models.last().unwrap();

            population = self.select(population, offspring, model, rng);
            fmin = find_fmin(all_evalutions.as_slice(), model).expect("fmin could be found");
        }

        let (suggestion_location, suggestion_statistics) = self.suggest_optimum(
            all_models.last().unwrap(),
            all_evalutions.as_slice(),
            rng,
            output,
            generation + 1,
        );

        Ok(OptimizationResult {
            all_individuals: all_evalutions,
            all_models,
            duration: total_duration.elapsed(),
            suggestion_location,
            suggestion_statistics,
        })
    }

    fn make_initial_population(&self, rng: &mut RNG) -> Vec<Individual<A>> {
        self.space.sample_individual_n(self.config.initial, rng)
    }

    fn evaluate_all(
        &self,
        individuals: &mut [Individual<A>],
        rng: &mut RNG,
        output: &mut Output<A>,
        generation: u16,
    ) {
        let timer = self.time_source.as_ref()();
        let objective = self.objective;

        let rngs = individuals
            .iter()
            .map(|_| rng.fork_random_state())
            .collect_vec();

        use rayon::prelude::*;

        individuals
            .par_iter_mut()
            .zip(rngs)
            .for_each(|(ind, mut rng): (&mut Individual<A>, _)| {
                let (observation, cost) = objective.run(ind.sample(), &mut rng);
                ind.set_evaluation_result(generation.into(), observation, cost)
                    .expect("individual cannot be evaluated twice");
                assert!(ind.is_fully_initialized());
            });

        output.event_evaluations_completed(individuals, timer.elapsed());
    }

    fn fit_next_model(
        &self,
        individuals: &[Individual<A>],
        gen: usize,
        prev_model: Option<&Estimator::Model>,
        rng: &mut RNG,
        output: &mut Output<A>,
    ) -> Result<Estimator::Model, Estimator::Error> {
        assert!(
            !individuals.is_empty(),
            "fit_next_model() requires at least one individual"
        );
        let timer = self.time_source.as_ref()();

        use ndarray::prelude::*;

        let xs = self.space.project_into_features_array(
            individuals
                .iter()
                .map(|ind| ind.sample())
                .collect_vec()
                .as_slice(),
        );

        let ys: Array1<A> = individuals
            .iter()
            .map(|ind| ind.observation().expect("individual has observation"))
            .collect();

        let model = self.estimator.estimate(xs, ys, prev_model, rng)?;

        output.event_model_trained(gen, &model, timer.elapsed());

        Ok(model)
    }

    fn resize_population(
        &self,
        mut population: Vec<Individual<A>>,
        newsize: usize,
        model: &Estimator::Model,
        rng: &mut RNG,
    ) -> Vec<Individual<A>> {
        // Sort the individuals.
        let fitness_operator = FitnessOperator(model, &self.space, self.config.select_via);
        population.sort_by(|a, b| {
            fitness_operator
                .compare(a, b)
                .expect("individuals are comparable")
        });
        // If there are too many individuals, remove worst-ranking individuals.
        population.truncate(newsize);

        // If there are too few inidividuals, repeat individusals.
        // It might be better to pad with random choices,
        // but these individuals must be fully evaluated.
        while population.len() < newsize {
            let i = rng.uniform(0..population.len());
            population.push(population[i].clone());
        }

        population
    }

    fn relscale_at_gen(&self, generation: i32) -> f64 {
        assert!(
            generation >= 1,
            "generation must be positive: {}",
            generation
        );
        let attenuation = self.config.relscale_attenuation.powi(generation - 1);
        attenuation * self.config.relscale_initial
    }

    fn acquire(
        &self,
        population: &[Individual<A>],
        model: &Estimator::Model,
        rng: &mut RNG,
        output: &mut Output<A>,
        fmin: A,
        relscale: &[f64],
    ) -> Vec<Individual<A>> {
        let timer = self.time_source.as_ref()();

        let offspring =
            self.acquisition_strategy
                .acquire(population, model, &self.space, rng, fmin, relscale);

        output.event_acquisition_completed(timer.elapsed());

        offspring
    }

    fn select(
        &self,
        parents: Vec<Individual<A>>,
        offspring: Vec<Individual<A>>,
        model: &Estimator::Model,
        rng: &mut RNG,
    ) -> Vec<Individual<A>> {
        let fitness_operator = FitnessOperator(model, &self.space, self.config.select_via);

        let (selected, rejected) = select_next_population(parents, offspring, fitness_operator);

        let n_second_chance = rng.binom(self.config.popsize, self.config.competition_rate);
        replace_worst_n_individuals(n_second_chance, selected, rejected, |a, b| {
            fitness_operator
                .compare(a, b)
                .expect("individuals should be orderable")
        })
    }

    fn suggest_optimum<'a>(
        &'a self,
        model: &'a Estimator::Model,
        individuals: &'a [Individual<A>],
        rng: &mut RNG,
        output: &mut Output<A>,
        gen: u16,
    ) -> (Vec<ParameterValue>, SummaryStatistics<A>) {
        let confidence_bound = A::from_f(self.config.confidence_bound);

        let (suggestion, suggestion_y) = find_best_individual_by_confidence_bound(
            individuals,
            model,
            &self.space,
            confidence_bound,
        );

        // We now have the minimum evaluated point.
        // Use that as a starting point for minimization of the response surface.
        let mut suggestion_features: Vec<f64> = self.space.project_into_features(suggestion);
        crate::util::minimize_without_gradient(
            |x, maybe_grad, _data| {
                assert!(maybe_grad.is_none(), "cannot provide gradients");
                let x_design = self.space.project_from_features(x);
                let x_normalized = self.space.project_into_features(x_design.as_slice());
                model
                    .predict_confidence_bound(x_normalized.into(), confidence_bound)
                    .into()
            },
            suggestion_features.as_mut_slice(),
            vec![(0.0, 1.0); self.space.len()].as_slice(),
            (),
        );

        let suggestion = self
            .space
            .project_from_features(suggestion_features.clone());

        let (suggestion_prediction, suggestion_ei) = model.predict_mean_ei(
            self.space
                .project_into_features(suggestion.as_slice())
                .into(),
            suggestion_y,
        );

        let mut validation_sample = Individual::new(suggestion.clone());
        validation_sample
            .set_prediction_and_ei(suggestion_prediction, suggestion_ei)
            .expect("individual cannot have previous prediction");
        let mut validation_samples = vec![validation_sample; self.config.validation];

        self.evaluate_all(validation_samples.as_mut_slice(), rng, output, gen);

        // combine new data with model, without refitting
        let individuals = individuals.iter().chain(&validation_samples).collect_vec();
        let xs = self.space.project_into_features_array(
            individuals
                .iter()
                .map(|ind| ind.sample())
                .collect_vec()
                .as_slice(),
        );
        let ys: Array1<A> = individuals
            .iter()
            .map(|ind| ind.observation().expect("individual has observation"))
            .collect();
        let newmodel = self.estimator.extend(xs, ys, model, rng);
        let model = newmodel.as_ref().unwrap_or(model);

        let suggestion_statistics =
            model.predict_statistics(Array1::from(suggestion_features).mapv(A::from_f));
        (suggestion, suggestion_statistics)
    }
}

#[derive(Clone, Copy)]
struct FitnessOperator<'life, A>(&'life dyn SurrogateModel<A>, &'life Space, FitnessVia);

impl<'life, A> FitnessOperator<'life, A> {
    fn get_fitness(&self, ind: &Individual<A>) -> Option<A>
    where
        A: Scalar,
    {
        let &FitnessOperator(ref model, ref space, fitness_via) = self;
        match fitness_via {
            FitnessVia::Prediction => {
                Some(model.predict_mean(space.project_into_features(ind.sample()).to_vec().into()))
            }
            FitnessVia::Observation => ind.observation(),
        }
    }

    fn compare(&self, a: &Individual<A>, b: &Individual<A>) -> Option<std::cmp::Ordering>
    where
        A: PartialOrd + Scalar,
    {
        let a = self.get_fitness(a);
        let b = self.get_fitness(b);
        a.partial_cmp(&b)
    }
}

fn find_best_individual_by_confidence_bound<'a, A, Model>(
    individuals: &'a [Individual<A>],
    model: &Model,
    space: &Space,
    confidence_bound: A,
) -> (&'a [ParameterValue], A)
where
    A: Scalar,
    Model: SurrogateModel<A>,
{
    let mut individuals_iter = individuals.iter();

    let mut suggestion = individuals_iter
        .next()
        .expect("should have at least one individual")
        .sample();
    let mut suggestion_ucb = model.predict_confidence_bound(
        space.project_into_features(suggestion).into(),
        confidence_bound,
    );

    for ind in individuals_iter {
        let candidate_ucb = model.predict_confidence_bound(
            space.project_into_features(ind.sample()).into(),
            confidence_bound,
        );
        if candidate_ucb < suggestion_ucb {
            suggestion = ind.sample();
            suggestion_ucb = candidate_ucb;
        }
    }

    let suggestion_y = model.predict_mean(space.project_into_features(suggestion).into());
    (suggestion, suggestion_y)
}

/// Select the offspring, unless the parent was better.
/// This avoids moving into “worse” areas,
/// although i doesn't consider the acquisition strategy (EI).
/// So this is a greey hill-climbing approach based purely
/// on the observed (possibly noisy) fitness value.
///
/// Each offspring only competes against its one parent!
fn select_next_population<A>(
    parents: Vec<Individual<A>>,
    offspring: Vec<Individual<A>>,
    fitness: FitnessOperator<A>,
) -> (Vec<Individual<A>>, Vec<Individual<A>>)
where
    A: Scalar,
{
    let mut selected = Vec::new();
    let mut rejected = Vec::new();

    for (parent, offspring) in parents.into_iter().zip_eq(offspring) {
        let (select, reject) =
            if fitness.compare(&parent, &offspring) == Some(std::cmp::Ordering::Less) {
                (parent, offspring)
            } else {
                (offspring, parent)
            };

        selected.push(select);
        rejected.push(reject);
    }

    (selected, rejected)
}

/// Allow the n worst individuals from the replacement pool
/// to become part of the population,
/// if they have better fitness.
fn replace_worst_n_individuals<T, Comparator>(
    n_replacements: usize,
    mut population: Vec<T>,
    mut replacement_pool: Vec<T>,
    comparator: Comparator,
) -> Vec<T>
where
    Comparator: Fn(&T, &T) -> std::cmp::Ordering,
{
    // find candidate replacements
    replacement_pool.sort_by(&comparator);
    replacement_pool.truncate(n_replacements);

    // select population members
    let population_size = population.len();
    population.extend(replacement_pool);
    population.sort_by(&comparator);
    population.truncate(population_size);
    population
}

#[cfg(test)]
mod fn_replace_worst_n_individuals {
    use super::*;

    #[test]
    fn replaces_worst_individuals() {
        assert_eq!(
            replace_worst_n_individuals(
                3,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![1, 2, 3],
                std::cmp::Ord::cmp
            ),
            vec![1, 1, 2, 2, 3, 3, 4, 5]
        );
    }

    #[test]
    fn does_not_replace_better_individuals() {
        assert_eq!(
            replace_worst_n_individuals(
                3,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![7, 8, 9, 10],
                std::cmp::Ord::cmp
            ),
            vec![1, 2, 3, 4, 5, 6, 7, 7]
        );
    }

    #[test]
    fn considers_up_to_n_replacements() {
        assert_eq!(
            replace_worst_n_individuals(
                3,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![2, 2, 2, 3],
                std::cmp::Ord::cmp
            ),
            vec![1, 2, 2, 2, 2, 3, 4, 5]
        );
    }
}
