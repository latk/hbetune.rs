use ndarray::prelude::*;

pub struct Individual<A> {
    pub sample: Array1<A>,
    pub gen: Option<isize>,
    pub observation: Option<A>,
}

impl<A> Individual<A> {
    pub fn new(sample: Array1<A>) -> Self {
        Individual {
            sample,
            gen: None,
            observation: None,
        }
    }

    pub fn is_fully_initialized(&self) -> bool {
        true
            && self.gen.is_some()
            && self.observation.is_some()
    }
}
