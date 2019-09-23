#[macro_use]
extern crate ndarray;
extern crate assert_cmd;
extern crate ggtune;
extern crate itertools;
extern crate noisy_float;
extern crate serde_json;

use assert_cmd::prelude::*;
use ggtune::{EstimatorGPR, ObjectiveFunctionFromFn, ParameterValue};
use itertools::Itertools as _;
use ndarray::prelude::*;
use serde_json::json;
use std::process::Command;

fn sphere_objective<A: ggtune::Scalar>(xs: &[ParameterValue]) -> A {
    let xs_array = xs.iter().cloned().map(Into::into).collect_vec().into();
    let y: f64 = ggtune::benchfn::sphere(xs_array);
    A::from_f(y)
}

#[test]
fn sphere_d2_f64() {
    run_minimize_test(
        Types::<f64, EstimatorGPR>::default(),
        ObjectiveFunctionFromFn::new(sphere_objective),
        582_347,
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
        582_347,
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
        1_098_438,
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
        98_482,
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
        64_026,
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
fn sphere_d2_noisy_integration() {
    run_integration_test(&[array![0.0, 0.0]], 0.2, |command| {
        command
            .arg("run")
            .arg("--seed=2956349")
            .arg("--max-nevals=70")
            .arg("--popsize=8")
            .arg("--initial=15")
            .arg("--param=x1 int -7 9")
            .arg("--param=x2 real -3 2")
            .arg("function")
            .arg("--noise=1")
            .arg("sphere");
    });
}

#[test]
fn abs_d2_log_noisy_integration() {
    run_integration_test(&[array![0.0, 1.0]], 0.2, |command| {
        command
            .arg("run")
            .arg("--seed=82348")
            .arg("--max-nevals=70")
            .arg("--popsize=8")
            .arg("--initial=15")
            .arg("--relscale-attenuation=0.85")
            .arg("--param=x1 logint -10 1000 11")
            .arg("--param=x2 logreal 1 2000")
            .arg("function")
            .arg("sum-abs")
            .arg("--noise=1.0");
    });
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

fn run_integration_test<Setup>(ideal: &[Array1<f64>], max_distance: f64, setup: Setup)
where
    Setup: Fn(&mut Command) -> (),
{
    let mut command = Command::cargo_bin("ggtune").unwrap();
    setup(&mut command);
    let output = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .output()
        .expect("test command failed to execute");

    assert!(
        output.status.success(),
        "test command must execute successfully but status was {}",
        output.status
    );

    let stdout = String::from_utf8(output.stdout).expect("test command output must be UTF-8");
    static RESULT_MARKER: &str = "optimization result: ";
    let result_location = stdout
        .find(RESULT_MARKER)
        .expect("output must contain result marker");
    let result_json = &stdout[result_location + RESULT_MARKER.len()..];
    let result: serde_json::Value =
        serde_json::from_str(result_json).expect("optimization result must be JSON");

    let location: &[_] = result["location"]
        .as_array()
        .expect("result location must be array");
    let distance = distance_to_ideal_points(
        &location
            .iter()
            .map(|x| x["value"].as_f64().expect("value must be a number"))
            .collect_vec()
            .into(),
        ideal,
    );

    if distance > max_distance {
        let mut data = result.clone();
        data["distance"] = json!(distance);
        data["max distance"] = json!(max_distance);
        data["ideal points"] = json!(ideal.iter().map(|x| x.to_vec()).collect_vec());
        panic!("optimal point was not found {:#}", data);
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

    let distance = distance_to_ideal_points(&guess, ideal);

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

fn distance_to_ideal_points(actual: &Array1<f64>, ideal_points: &[Array1<f64>]) -> f64 {
    ideal_points
        .iter()
        .map(|ideal| (ideal - actual).mapv(|x| x.powi(2)).sum().sqrt())
        .map(noisy_float::types::n64)
        .min()
        .expect("there should be a minimal distance")
        .into()
}
