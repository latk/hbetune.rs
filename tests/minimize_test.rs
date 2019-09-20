#[macro_use]
extern crate ndarray;
extern crate ggtune;
extern crate itertools;
extern crate noisy_float;

use ggtune::{EstimatorGPR, ObjectiveFunctionFromFn};
use itertools::Itertools as _;
use ndarray::prelude::*;

fn sphere_objective<A: ggtune::Scalar>(xs: &[ggtune::ParameterValue]) -> A {
    let xs_array = xs.iter().cloned().map(Into::into).collect_vec().into();
    let y: f64 = ggtune::benchfn::sphere(xs_array);
    A::from_f(y)
}

#[test]
fn sphere_d2_f64() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        904827189,
        &[array![0.0, 0.0]],
        0.2,
        |space, minimizer, _args| {
            minimizer.max_nevals = 40;
            space.add_real_parameter("x1", -2.0, 3.0);
            space.add_real_parameter("x2", -3.0, 2.0);
        },
    );
}

#[test]
fn sphere_d2_f32() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        1759340364,
        &[array![0.0, 0.0]],
        0.2,
        |space, minimizer, _args| {
            minimizer.max_nevals = 40;
            space.add_real_parameter("x1", -2.0, 3.0);
            space.add_real_parameter("x2", -3.0, 2.0);
        },
    );
}

#[test]
fn sphere_d2_edge() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        1098438,
        &[array![0.1, 0.0]],
        0.2,
        |space, minimizer, _args| {
            minimizer.max_nevals = 50;
            space.add_real_parameter("x1", 0.1, 4.0);
            space.add_real_parameter("x2", -2.0, 2.0);
        },
    );
}

struct Types<A, Model> {
    marker: std::marker::PhantomData<(A, Model)>,
}

impl<A, Model> Default for Types<A, Model> {
    fn default() -> Self {
        Self {
            marker: std::marker::PhantomData::default(),
        }
    }
}

fn run_minimize_test<A, Model, ObjectiveFn, SetupFn>(
    _types: Types<A, Model>,
    objective: ObjectiveFn,
    rng_seed: usize,
    ideal: &[Array1<f64>],
    max_distance: f64,
    setup: SetupFn,
) where
    A: ggtune::Scalar,
    Model: ggtune::Estimator<A>,
    ObjectiveFn: ggtune::ObjectiveFunction<A>,
    SetupFn: Fn(&mut ggtune::Space, &mut ggtune::Minimizer, &mut ggtune::MinimizerArgs<A, Model>),
{
    assert!(!ideal.is_empty(), "a slice of ideal points is required");

    let mut rng = ggtune::RNG::new_with_seed(rng_seed);

    let mut minimizer = ggtune::Minimizer::default();
    let mut space = ggtune::Space::new();
    let mut args = ggtune::MinimizerArgs::default();
    setup(&mut space, &mut minimizer, &mut args);
    args.output
        .add_human_readable_individuals(std::io::stderr(), &space);

    let result = minimizer
        .minimize(&objective, space, &mut rng, args)
        .expect("minimization should proceed successfully");

    let guess: Array1<f64> = result
        .best_individual()
        .expect("there should be a best individual")
        .sample()
        .iter()
        .cloned()
        .map(Into::into)
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
