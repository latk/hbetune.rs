use crate::{Individual, Scalar, Space, SurrogateModel, RNG};
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
                let parent_sample = parent.sample();

                let candidate_samples = std::iter::repeat_with(|| {
                    let mut sample = parent_sample.to_vec();
                    space.mutate_inplace(&mut sample, relscale, rng);
                    sample
                })
                .take(self.breadth)
                .collect_vec();

                let (candidate_mean, candidate_ei): (Vec<A>, Vec<A>) = candidate_samples
                    .iter()
                    .map(|candidate| {
                        model.predict_mean_ei(space.project_into_features(candidate).into(), fmin)
                    })
                    .unzip();

                let i: usize = (0..candidate_ei.len())
                    .max_by(|&a, &b| {
                        candidate_ei[a]
                            .partial_cmp(&candidate_ei[b])
                            .expect("EI should be comparable")
                    })
                    .expect("there should be a candidate with maximal EI");

                let sample = take_vec_item(candidate_samples, i);
                let mut offspring = Individual::new(sample);
                offspring
                    .set_prediction_and_ei(candidate_mean[i], A::from_f(candidate_ei[i]))
                    .expect("individual cannot have previous prediction");
                offspring
            })
            .collect()
    }
}

pub fn expected_improvement(mean: f64, std: f64, fmin: f64) -> f64 {
    use statrs::distribution::{Continuous, Univariate};
    let norm = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let z = -(mean - fmin) / std;
    -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z)
}

fn take_vec_item<T>(mut xs: Vec<T>, i: usize) -> T {
    xs.swap_remove(i)
}
