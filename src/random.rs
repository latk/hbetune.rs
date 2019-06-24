extern crate rand_core;
extern crate rand;
extern crate rand_xoshiro;
// use rand_xoshiro::rand_core::SeedableRng;
use rand_core::{SeedableRng, RngCore};
use rand::Rng;
use rand::distributions::{self, Distribution};

type BasicRNG = rand_xoshiro::Xoshiro256StarStar;

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
}
