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
                            .ok_or_else(|| {
                                format!(
                                    "EI should be comparable: a={} b={}",
                                    candidate_ei[a], candidate_ei[b]
                                )
                            })
                            .unwrap()
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
