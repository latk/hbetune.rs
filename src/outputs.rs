use crate::{Individual, Space, SurrogateModel};
use std::boxed::Box;
use std::time::Duration;

/// Report progress and save results during optimization process.
pub trait OutputEventHandler<A> {
    /// Called when a new generation is started.
    fn event_new_generation(&mut self, _gen: usize, _relscale: &[f64]) {}

    /// Called when evaluations of a generation has completed.
    fn event_evaluations_completed(&mut self, _individuals: &[Individual<A>], _duration: Duration) {
    }

    /// Called when a new model has been trained
    fn event_model_trained(
        &mut self,
        _gen: usize,
        _model: &dyn SurrogateModel<A>,
        _duration: Duration,
    ) {
    }

    /// Called when new samples have been acquired.
    fn event_acquisition_completed(&mut self, _duration: Duration) {}
}

pub struct CompositeOutputEventHandler<A> {
    subloggers: Vec<Box<dyn OutputEventHandler<A>>>,
}

impl<A> CompositeOutputEventHandler<A> {
    pub fn new() -> Self {
        let subloggers = Vec::new();
        Self { subloggers }
    }

    pub fn add(&mut self, logger: impl Into<Box<dyn OutputEventHandler<A>>>) {
        self.subloggers.push(logger.into());
    }
}

impl<A> std::default::Default for CompositeOutputEventHandler<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> OutputEventHandler<A> for CompositeOutputEventHandler<A> {
    fn event_new_generation(&mut self, gen: usize, relscale: &[f64]) {
        for logger in &mut self.subloggers {
            logger.event_new_generation(gen, relscale);
        }
    }

    fn event_evaluations_completed(&mut self, individuals: &[Individual<A>], duration: Duration) {
        for logger in &mut self.subloggers {
            logger.event_evaluations_completed(individuals, duration);
        }
    }

    fn event_model_trained(
        &mut self,
        gen: usize,
        model: &dyn SurrogateModel<A>,
        duration: Duration,
    ) {
        for logger in &mut self.subloggers {
            logger.event_model_trained(gen, model, duration);
        }
    }

    fn event_acquisition_completed(&mut self, duration: Duration) {
        for logger in &mut self.subloggers {
            logger.event_acquisition_completed(duration);
        }
    }
}

pub struct Output<A> {
    base: CompositeOutputEventHandler<A>,
    evaluation_durations: Vec<Duration>,
    training_durations: Vec<Duration>,
    acquisition_durations: Vec<Duration>,
}

impl<A> Output<A> {
    pub fn new(_space: &Space) -> Self {
        let base = CompositeOutputEventHandler::new();
        Output {
            base,
            evaluation_durations: Vec::new(),
            training_durations: Vec::new(),
            acquisition_durations: Vec::new(),
        }
    }

    pub fn evaluation_durations(&self) -> &[Duration] {
        &self.evaluation_durations
    }

    pub fn training_durations(&self) -> &[Duration] {
        &self.training_durations
    }

    pub fn acquisition_durations(&self) -> &[Duration] {
        &self.acquisition_durations
    }
}

impl<A> OutputEventHandler<A> for Output<A> {
    fn event_new_generation(&mut self, gen: usize, relscale: &[f64]) {
        self.base.event_new_generation(gen, relscale);
    }

    fn event_evaluations_completed(&mut self, individuals: &[Individual<A>], duration: Duration) {
        self.evaluation_durations.push(duration);
        self.base.event_evaluations_completed(individuals, duration);
    }

    fn event_model_trained(
        &mut self,
        generation: usize,
        model: &dyn SurrogateModel<A>,
        duration: Duration,
    ) {
        self.training_durations.push(duration);
        self.base.event_model_trained(generation, model, duration);
    }

    fn event_acquisition_completed(&mut self, duration: Duration) {
        self.acquisition_durations.push(duration);
        self.base.event_acquisition_completed(duration);
    }
}
