#[macro_use]
extern crate ndarray;
extern crate ggtune;
extern crate itertools;
extern crate noisy_float;

use ggtune::{EstimatorGPR, ObjectiveFunctionFromFn, ParameterValue};
use itertools::Itertools as _;
use ndarray::prelude::*;

fn sphere_objective<A: ggtune::Scalar>(xs: &[ParameterValue]) -> A {
    let xs_array = xs.iter().cloned().map(Into::into).collect_vec().into();
    let y: f64 = ggtune::benchfn::sphere(xs_array);
    A::from_f(y)
}

struct NoisySphereObjective {
    sigma: f64,
}

impl<A: ggtune::Scalar> ggtune::ObjectiveFunction<A> for NoisySphereObjective {
    fn run(&self, xs: &[ParameterValue], rng: &mut ggtune::RNG) -> (A, A) {
        let y: f64 = sphere_objective(xs);
        (A::from_f(rng.normal(y, self.sigma)), A::from_i(0))
    }
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

#[test]
fn sphere_d2_ints() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        98482,
        &[array![0.0, 0.0]],
        1e-5,
        |space, minimizer, _args| {
            minimizer.max_nevals = 30;
            minimizer.popsize = 5;
            space.add_integer_parameter("x1", -5, 5);
            space.add_integer_parameter("x2", -3, 7);
        },
    );
}

#[test]
fn sphere_d2_mixed_int_float() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        64026,
        &[array![0.0, 0.0]],
        0.05,
        |space, minimizer, _args| {
            minimizer.max_nevals = 40;
            minimizer.popsize = 8;
            space.add_integer_parameter("x1", -5, 15);
            space.add_real_parameter("x2", -3.0, 2.0);
        },
    );
}

#[test]
fn sphere_d2_noisy() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        NoisySphereObjective { sigma: 1.0 },
        2956349,
        &[array![0.0, 0.0]],
        0.2,
        |space, minimizer, _args| {
            minimizer.max_nevals = 70;
            minimizer.popsize = 8;
            minimizer.initial = 15;
            space.add_integer_parameter("x1", -7, 9);
            space.add_real_parameter("x2", -3.0, 2.0);
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
    // args.output
    //     .add_human_readable_individuals(std::io::stderr(), &space);

    let result = minimizer
        .minimize(&objective, space, &mut rng, args)
        .expect("minimization should proceed successfully");

    let (value, value_std) = result.suggestion_y_std();

    let guess: Array1<f64> = result
        .suggestion()
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
         |        value: {value} ± {value_std}\n\
         |      history: {history:.2}",
        distance = distance,
        ideal = ideal.iter().format(", "),
        guess = guess,
        value = value,
        value_std = value_std,
        history = result
            .ys()
            .expect("there should be results")
            .iter()
            .format(", "),
    );
}
