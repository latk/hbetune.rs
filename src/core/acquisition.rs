use crate::{Individual, ParameterValue, Scalar, Space, SurrogateModel, RNG};
use itertools::Itertools as _;

pub trait AcquisitionStrategy<A> {
    /// Acquire new individuals. Should return `population.len()` new individuals.
    ///
    /// * population: previous population / parents
    /// * model: current model of the utility landscape
    /// * relscale: suggested normalized standard deviation for mutating individuals
    /// * fmin: current best observed value (useful for EI)
    fn acquire(
        &self,
        population: &[Individual<A>],
        model: &dyn SurrogateModel<A>,
        space: &Space,
        rng: &mut RNG,
        fmin: A,
        relscale: &[f64],
    ) -> Vec<Individual<A>>
    where
        A: Scalar;
}

/// Uses a gradient-free search for the best EI.
/// Only *one* parent is used for this, the others use the BaseAcquisition.
pub struct SearchingAcquisition<BaseAcquisition> {
    pub base: BaseAcquisition,
}

impl<A, BaseAcquisition> AcquisitionStrategy<A> for SearchingAcquisition<BaseAcquisition>
where
    BaseAcquisition: AcquisitionStrategy<A>,
{
    fn acquire(
        &self,
        population: &[Individual<A>],
        model: &dyn SurrogateModel<A>,
        space: &Space,
        rng: &mut RNG,
        fmin: A,
        relscale: &[f64],
    ) -> Vec<Individual<A>>
    where
        A: Scalar,
    {
        let candidate_samples = population
            .iter()
            .map(|parent| {
                let mut candidate = space.project_into_features(parent.sample());
                crate::util::minimize_without_gradient(
                    |x, maybe_grad, _data| {
                        assert!(maybe_grad.is_none(), "cannot provide gradients");
                        let x_design = space.project_from_features(x);
                        let x_normalized = space.project_into_features(x_design.as_slice());
                        let (_mean, ei) = model.predict_mean_ei(x_normalized.into(), fmin);
                        -ei.into()
                    },
                    candidate.as_mut_slice(),
                    vec![(0., 1.); space.len()].as_slice(),
                    (),
                );
                space.project_from_features(candidate)
            })
            .collect_vec();

        let (i, mean, ei) = find_best_candidate_by_ei(&candidate_samples, model, space, fmin);
        let mut offspring = Individual::new(take_vec_item(candidate_samples, i));
        offspring
            .set_prediction_and_ei(mean, ei)
            .expect("individual cannot have previous prediction");

        // delegate acquisition for other parents to BaseAcquisition
        let mut other_offspring = self
            .base
            .acquire(population, model, space, rng, fmin, relscale);
        other_offspring[i] = offspring;

        other_offspring
    }
}

pub struct MutationAcquisition {
    pub breadth: usize,
}

impl<A> AcquisitionStrategy<A> for MutationAcquisition {
    fn acquire(
        &self,
        population: &[Individual<A>],
        model: &dyn SurrogateModel<A>,
        space: &Space,
        rng: &mut RNG,
        fmin: A,
        relscale: &[f64],
    ) -> Vec<Individual<A>>
    where
        A: Scalar,
    {
        population
            .iter()
            .map(|parent| {
                let candidate_samples = self.generate_nearby_samples(parent, space, rng, relscale);

                let (i, mean, ei) =
                    find_best_candidate_by_ei(&candidate_samples, model, space, fmin);

                let sample = take_vec_item(candidate_samples, i);
                let mut offspring = Individual::new(sample);
                offspring
                    .set_prediction_and_ei(mean, ei)
                    .expect("individual cannot have previous prediction");
                offspring
            })
            .collect()
    }
}

impl MutationAcquisition {
    fn generate_nearby_samples<A>(
        &self,
        parent: &Individual<A>,
        space: &Space,
        rng: &mut RNG,
        relscale: &[f64],
    ) -> Vec<Vec<ParameterValue>>
    where
        A: Scalar,
    {
        let parent_sample = parent.sample();

        std::iter::repeat_with(|| {
            let mut sample = parent_sample.to_vec();
            space.mutate_inplace(&mut sample, relscale, rng);
            sample
        })
        .take(self.breadth)
        .collect_vec()
    }
}

pub fn expected_improvement(mean: f64, std: f64, fmin: f64) -> f64 {
    assert!(mean.is_finite(), "mean must be finite: {}", mean);
    assert!(std.is_finite(), "std must be finite: {}", std);
    assert!(fmin.is_finite(), "fmin must be finite: {}", fmin);

    // trivial case: if std is zero, the EI depends purely on the position of the mean.
    // That way, we don't have to calculate z-scores which could become NaN.
    if std <= 0.0 || ulps_eq!(std, 0.0) {
        if mean < fmin {
            // expected improvement by the difference is guaranteed
            return -(mean - fmin);
        } else {
            // guaranteed that no improvement is possible
            return 0.0;
        }
    }

    // do the full calculations using z-scores
    use statrs::distribution::{Continuous, Univariate};
    let norm = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let z = -(mean - fmin) / std;
    let ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z);

    assert!(ei.is_finite(), "EI must be finite: {}", ei);
    assert!(
        ei >= 0.0 || ulps_eq!(ei, 0.0),
        "EI must be non-negative: {}",
        ei
    );
    ei
}

fn take_vec_item<T>(mut xs: Vec<T>, i: usize) -> T {
    xs.swap_remove(i)
}

fn find_best_candidate_by_ei<A>(
    candidates: &[Vec<ParameterValue>],
    model: &dyn SurrogateModel<A>,
    space: &Space,
    fmin: A,
) -> (usize, A, A)
where
    A: Scalar,
{
    let (means, eis): (Vec<A>, Vec<A>) = candidates
        .iter()
        .map(|candidate| model.predict_mean_ei(space.project_into_features(candidate).into(), fmin))
        .unzip();

    let i: usize = (0..eis.len())
        .max_by(|&a, &b| {
            if let Some(cmp) = eis[a].partial_cmp(&eis[b]) {
                cmp
            } else {
                panic!("EI should be comparable: a={} b={}", eis[a], eis[b])
            }
        })
        .expect("there should be a candidate with maximal EI");

    (i, means[i], eis[i])
}
