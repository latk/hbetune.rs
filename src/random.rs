extern crate rand_core;
extern crate rand;
extern crate rand_xoshiro;
// use rand_xoshiro::rand_core::SeedableRng;
use rand_core::SeedableRng;
use rand::distributions::{self, Distribution as _};

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

    pub fn basic_rng_mut(&mut self) -> &mut BasicRNG { &mut self.basic_rng }

    pub fn uniform<T>(&mut self, lo: T, hi: T) -> T
    where T: rand::distributions::uniform::SampleUniform
    {
        distributions::Uniform::new_inclusive(lo, hi).sample(self.basic_rng_mut())
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        distributions::Normal::new(mean, std).sample(self.basic_rng_mut())
    }
}
