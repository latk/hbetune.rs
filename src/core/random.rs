extern crate rand;
extern crate rand_core;
extern crate rand_xoshiro;
use rand::distributions::{self, Distribution as _};
use rand_core::SeedableRng;

type BasicRNG = rand_xoshiro::Xoshiro256StarStar;

#[derive(Debug, Clone)]
pub struct RNG {
    basic_rng: BasicRNG,
}

impl RNG {
    pub fn new_with_seed(seed: usize) -> Self {
        let basic_rng = SeedableRng::seed_from_u64(seed as u64);
        RNG { basic_rng }
    }

    pub fn fork_random_state(&mut self) -> Self {
        let basic_rng = SeedableRng::from_rng(&mut self.basic_rng).unwrap();
        RNG { basic_rng }
    }

    pub fn basic_rng_mut(&mut self) -> &mut BasicRNG {
        &mut self.basic_rng
    }

    pub fn uniform<T, Range>(&mut self, range: Range) -> T
    where
        Range: Into<rand::distributions::Uniform<T>>,
        T: rand::distributions::uniform::SampleUniform,
    {
        range.into().sample(self.basic_rng_mut())
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        distributions::Normal::new(mean, std).sample(self.basic_rng_mut())
    }
}
