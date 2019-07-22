use crate::{Individual, SurrogateModel, Space, RNG};

pub trait AcquisitionStrategy<A> {
    fn acquire(
        &self,
        population: &[Individual<A>],
        model: &dyn SurrogateModel<A>,
        space: &Space,
        rng: &mut RNG,
        fmin: A,
        relscale: &[f64],
    ) -> Vec<Individual<A>>;
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
    ) -> Vec<Individual<A>> {
        vec![]
    }
}
