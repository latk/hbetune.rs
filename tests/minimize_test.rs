#[macro_use]
extern crate ndarray;
extern crate ggtune;
extern crate itertools;
extern crate noisy_float;

use ggtune::{EstimatorGPR, Minimizer};
use itertools::Itertools as _;
use ndarray::prelude::*;

#[test]
fn sphere_d2_f64() {
    run_minimize_test(
        |xs: ArrayView1<_>| ggtune::benchfn::sphere(xs),
        904827189,
        &[array![0.0, 0.0]],
        0.2,
        |minimizer: &mut Minimizer<f64, EstimatorGPR>, space| {
            minimizer.max_nevals = 40;
            space.add_real_parameter("x1", -2.0, 3.0);
            space.add_real_parameter("x2", -3.0, 2.0);
        },
    );
}

#[test]
fn sphere_d2_f32() {
    run_minimize_test(
        |xs: ArrayView1<_>| ggtune::benchfn::sphere(xs),
        37895438,
        &[array![0.0, 0.0]],
        0.2,
        |minimizer: &mut Minimizer<f64, EstimatorGPR>, space| {
            minimizer.max_nevals = 40;
            space.add_real_parameter("x1", -2.0, 3.0);
            space.add_real_parameter("x2", -3.0, 2.0);
        },
    );
}

#[test]
fn sphere_d2_edge() {
    run_minimize_test(
        |xs: ArrayView1<_>| ggtune::benchfn::sphere(xs),
        8345729,
        &[array![0.1, 0.0]],
        0.2,
        |minimizer: &mut Minimizer<f64, EstimatorGPR>, space| {
            minimizer.max_nevals = 50;
            space.add_real_parameter("x1", 0.1, 4.0);
            space.add_real_parameter("x2", -2.0, 2.0);
        },
    );
}

fn run_minimize_test<A, Model, SetupFn>(
    objective: impl ggtune::ObjectiveFunction<A>,
    rng_seed: usize,
    ideal: &[Array1<A>],
    max_distance: f64,
    setup: SetupFn,
) where
    A: ggtune::Scalar,
    Model: ggtune::Estimator<A>,
    SetupFn: Fn(&mut ggtune::Minimizer<A, Model>, &mut ggtune::Space),
{
    assert!(!ideal.is_empty(), "a slice of ideal points is required");

    let mut rng = ggtune::RNG::new_with_seed(rng_seed);

    let mut minimizer = ggtune::Minimizer::<A, Model>::default();
    let mut space = ggtune::Space::new();
    setup(&mut minimizer, &mut space);

    let result = minimizer
        .minimize(&objective, space, &mut rng, None, vec![])
        .expect("minimization should proceed successfully");

    let guess: Array1<A> = result
        .best_individual()
        .expect("there should be a best individual")
        .sample()
        .iter()
        .cloned()
        .collect();

    let distance: f64 = ideal
        .iter()
        .map(|ideal| (ideal - &guess).mapv(|x| x.powi(2)).sum().sqrt())
        .map(ggtune::Scalar::to_n64)
        .min()
        .expect("there should be a minimal distance")
        .into();

    assert!(
        distance < max_distance,
        "optimal point was not found\n\
         |     distance: {distance}\n\
         | ideal points: {ideal}\n\
         |        guess: {guess}\n\
         |      history: {history:.2}",
        distance = distance,
        ideal = ideal.iter().format(", "),
        guess = guess,
        history = result
            .ys()
            .expect("there should be results")
            .iter()
            .format(", "),
    );
}
