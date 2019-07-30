use crate::{Individual, SurrogateModel, Space, RNG, Scalar};
use itertools::Itertools as _;
use ndarray::prelude::*;

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
    where A: Scalar;
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
    where A: Scalar
    {
        population.iter().map(|parent| {
            let parent_sample_transformed = space.into_transformed(parent.sample().to_owned());

            let candidate_samples: Vec<Array1<A>> = std::iter::from_fn(|| {
                Some(space.mutate_transformed(parent_sample_transformed.clone(), relscale, rng))
            }).take(self.breadth).collect();

            let (candidate_mean, candidate_std): (Vec<A>, Vec<A>) = candidate_samples.iter()
                .map(|candidate| model.predict_mean_std_transformed(candidate.to_owned()))
                .unzip();

            let candidate_ei: Vec<f64> = candidate_mean.iter().zip_eq(&candidate_std)
                .map(|(&mean, &std)| expected_improvement(mean.into(), std.into(), fmin.into()))
                .collect();

            let i: usize = (0 .. candidate_ei.len()).into_iter()
                .max_by(|&a, &b| candidate_ei[a].partial_cmp(&candidate_ei[b])
                        .expect("EI should be comparable"))
                .expect("there should be a candidate with maximal EI");

            let sample = space.from_transformed(take_vec_item(candidate_samples, i));
            let mut offspring = Individual::new(sample);
            offspring.set_prediction(candidate_mean[i]).unwrap();
            offspring.set_expected_improvement(A::from_f(candidate_ei[i])).unwrap();
            offspring
        }).collect()
    }
}

pub fn expected_improvement(mean: f64, std: f64, fmin: f64) -> f64
{
    use statrs::distribution::{Univariate, Continuous};
    let norm = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let z = -(mean - fmin) / std;
    let ei = -(mean - fmin) * norm.cdf(z) + std * norm.pdf(z);
    ei
}

fn take_vec_item<T>(mut xs: Vec<T>, i: usize) -> T {
    xs.swap_remove(i)
}
