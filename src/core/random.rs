extern crate rand;
extern crate rand_core;
extern crate rand_distr;
extern crate rand_xoshiro;
use rand_core::SeedableRng;
use rand_distr::{self as distributions, Distribution as _};
use std::convert::TryInto as _;

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
        Range: Into<distributions::Uniform<T>>,
        T: distributions::uniform::SampleUniform,
    {
        range.into().sample(self.basic_rng_mut())
    }

    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        distributions::Normal::new(mean, std)
            .expect("should get normal distribution")
            .sample(self.basic_rng_mut())
    }

    pub fn binom(&mut self, n: usize, p: f64) -> usize {
        distributions::Binomial::new(n.try_into().expect("n fits into u64"), p)
            .expect("should get binomial distribution")
            .sample(self.basic_rng_mut())
            .try_into()
            .expect("binom result fits into usize")
    }
}
