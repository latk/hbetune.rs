use crate::{AcquisitionStrategy, MutationAcquisition};
use crate::{Estimator as ModelEstimator, SurrogateModel};
use crate::{Individual, Output, OutputEventHandler as _, Scalar, Space, RNG};

use std::iter::FromIterator as _;
use std::time::{Duration, Instant};

use itertools::Itertools as _;
use ndarray::prelude::*;

type TimeSource = Fn() -> Instant;

pub trait ObjectiveFunction<A>: Sync {
    fn run<'a>(&self, xs: ArrayView1<'a, A>, rng: &'a mut RNG) -> (A, A);
}

pub struct ObjectiveFunctionFromFn<F>(F);

impl<F> ObjectiveFunctionFromFn<F> {
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<A, F> ObjectiveFunction<A> for ObjectiveFunctionFromFn<F>
where
    F: Fn(ArrayView1<A>) -> A + Sync,
    A: Default,
{
    fn run<'a>(&self, xs: ArrayView1<'a, A>, _rng: &'a mut RNG) -> (A, A) {
        let Self(f) = self;
        (f(xs), Default::default())
    }
}

pub struct OptimizationResult<A, Model> {
    all_individuals: Vec<Individual<A>>,
    all_models: Vec<Model>,
    duration: Duration,
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

    /// Select the best evaluation results.
    pub fn best_n(&self, n: usize) -> impl Iterator<Item = &Individual<A>>
    where
        A: PartialOrd + Copy,
    {
        self.all_individuals
            .iter()
            .filter(|ind| ind.observation().is_some())
            .sorted_by(|a, b| {
                a.observation()
                    .partial_cmp(&b.observation())
                    .expect("observation should have an order")
            })
            .take(n)
    }

    /// Best result.
    pub fn best_individual(&self) -> Option<&Individual<A>>
    where
        A: PartialOrd + Copy,
    {
        self.best_n(1).next()
    }

    /// Final model.
    pub fn model(&self) -> Option<&Model> {
        self.all_models.last()
    }

    /// Input variables of all evaluations
    pub fn xs(&self) -> Array2<A>
    where
        A: Copy,
    {
        let samples = self
            .all_individuals
            .iter()
            .map(|ind| ind.sample().insert_axis(Axis(0)))
            .collect_vec();
        ndarray::stack(Axis(0), &samples).unwrap()
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
        Some(Array::from_vec(ys?))
    }

    /// Best observed value.
    pub fn fmin(&self) -> Option<A>
    where
        A: Copy + PartialOrd,
    {
        self.best_individual().and_then(|ind| ind.observation())
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

    /// How many random samples are suggested per generation.
    /// Usually, new samples are created by random mutations of existing samples.
    #[structopt(long, default_value = "1")]
    pub n_replacements: usize,
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
            max_nevals: 100,
            relscale_initial: 0.3,
            relscale_attenuation: 0.9,
            select_via: FitnessVia::Observation,
            fmin_via: FitnessVia::Prediction,
            n_replacements: 1,
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
            self.initial + self.popsize <= self.max_nevals,
            "evaluation budget {max_nevals} too small with {initial}+n*{popsize} evaluations",
            max_nevals = self.max_nevals,
            initial = self.initial,
            popsize = self.popsize
        );

        let acquisition_strategy =
            acquisition_strategy.unwrap_or_else(|| Box::new(MutationAcquisition { breadth: 10 }));

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

        let mut budget = config.max_nevals;

        let mut all_evalutions: Vec<Individual<A>> = Vec::new();
        let mut all_models: Vec<Estimator::Model> = Vec::new();

        all_evalutions.extend(historic_individuals);

        for ind in &mut population {
            ind.set_prediction(A::from_f(0.0)).unwrap();
            ind.set_expected_improvement(A::from_f(0.0)).unwrap();
        }

        self.evaluate_all(population.as_mut_slice(), rng, output, 0);
        budget = budget.saturating_sub(population.len());
        all_evalutions.extend(population.iter().cloned());

        all_models.push(self.fit_next_model(all_evalutions.as_slice(), 0, None, rng, output)?);

        let find_fmin = |individuals: &[Individual<A>], model: &Estimator::Model| {
            let fmin_operator = FitnessOperator(model, config.fmin_via);
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

            population = self.select(population, offspring, model);
            fmin = find_fmin(all_evalutions.as_slice(), model).expect("fmin could be found");
        }

        Ok(OptimizationResult {
            all_individuals: all_evalutions,
            all_models,
            duration: total_duration.elapsed(),
        })
    }

    fn make_initial_population(&self, rng: &mut RNG) -> Vec<Individual<A>> {
        std::iter::from_fn(|| Some(self.space.sample(rng)))
            .take(self.config.initial)
            .collect_vec()
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
                ind.set_observation(observation)
                    .expect("observation was unset");
                ind.set_cost(cost).expect("cost was unset");
                ind.set_gen(generation.into())
                    .expect("generation was unset");
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
        let timer = self.time_source.as_ref()();

        use ndarray::prelude::*;

        let xs: Array2<A> = ndarray::stack(
            Axis(0),
            individuals
                .iter()
                .map(|ind| ind.sample().insert_axis(Axis(0)))
                .collect_vec()
                .as_slice(),
        )
        .unwrap();

        let ys: Array1<A> = individuals
            .iter()
            .map(|ind| ind.observation().expect("individual has observation"))
            .collect();

        let model = self
            .estimator
            .estimate(xs, ys, self.space.clone(), prev_model, rng)?;

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
        let fitness_operator = FitnessOperator(model, self.config.select_via);
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
    ) -> Vec<Individual<A>> {
        let fitness_operator = FitnessOperator(model, self.config.select_via);

        let (selected, rejected) = select_next_population(parents, offspring, fitness_operator);

        replace_worst_n_individuals(self.config.n_replacements, selected, rejected, |a, b| {
            fitness_operator
                .compare(a, b)
                .expect("individuals should be orderable")
        })
    }
}

#[derive(Clone, Copy)]
struct FitnessOperator<'life, A>(&'life SurrogateModel<A>, FitnessVia);

impl<'life, A> FitnessOperator<'life, A> {
    fn get_fitness(&self, ind: &Individual<A>) -> Option<A>
    where
        A: Scalar,
    {
        let &FitnessOperator(ref model, fitness_via) = self;
        match fitness_via {
            FitnessVia::Prediction => Some(model.predict_mean(ind.sample().to_owned())),
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
speculate::speculate! {
    describe "fn replace_worst_n_individuals()" {
        it "replaces worst individuals" {
            assert_eq!(
                replace_worst_n_individuals(
                    3, vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1, 2, 3],
                    std::cmp::Ord::cmp),
                vec![1, 1, 2, 2, 3, 3, 4, 5]);
        }

        it "does not replace better individuals" {
            assert_eq!(
                replace_worst_n_individuals(
                    3, vec![1, 2, 3, 4, 5, 6, 7, 8], vec![7, 8, 9, 10],
                    std::cmp::Ord::cmp),
                vec![1, 2, 3, 4, 5, 6, 7, 7]);
        }

        it "considers up to n replacements" {
            assert_eq!(
                replace_worst_n_individuals(
                    3, vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2, 3],
                    std::cmp::Ord::cmp),
                vec![1, 2, 2, 2, 2, 3, 4, 5]);
        }
    }
}
