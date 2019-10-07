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

#[test]
fn goldstein_price_integration() {
    run_integration_test(&[array![0.0, -1.0]], 0.05, |command| {
        command
            .arg("run")
            .arg("--seed=19741")
            .arg("--max-nevals=100")
            .arg("--initial=30")
            .arg("--popsize=7")
            .arg("--param=x1 real -2 2")
            .arg("--param=x2 real -2 2")
            .args(&["function", "goldstein-price"]);
    });
}

#[test]
fn goldstein_price_noisy_integration() {
    run_integration_test(&[array![0.0, -1.0]], 0.1, |command| {
        command
            .arg("run")
            .arg("--seed=32371")
            .arg("--max-nevals=114")
            .arg("--initial=30")
            .arg("--popsize=7")
            .arg("--param=x1 real -2 2")
            .arg("--param=x2 real -2 2")
            .args(&["function", "goldstein-price", "--noise=1.0"]);
    });
}

#[test]
fn goldstein_price_logy() {
    run_integration_test(&[array![0.0, -1.0]], 0.05, |command| {
        command
            .arg("run")
            .arg("--seed=2029")
            .arg("--max-nevals=79")
            .arg("--initial=30")
            .arg("--popsize=7")
            .arg("--transform-objective=log")
            .arg("--param=x1 real -2 2")
            .arg("--param=x2 real -2 2")
            .args(&["function", "goldstein-price"]);
    });
}

/// The Easom test doesn't really have any chance of heuristically finding the optimum,
/// so that we put most points into exploration
/// and allow any recommendation point as “correct” result.
#[test]
fn easom_integration() {
    use std::f64::consts::PI;
    run_integration_test(&[array![PI, PI]], 500.0, |command| {
        command
            .arg("run")
            .arg("--seed=426")
            .arg("--max-nevals=100")
            .arg("--initial=90")
            .arg("--popsize=5")
            .arg("--param=x1 real -50 50")
            .arg("--param=x2 real -50 50")
            .args(&["function", "easom"]);
    });
}

fn himmelblau_ideal() -> [Array1<f64>; 4] {
    [
        array![3.0, 2.0],
        array![-2.805118, 3.131312],
        array![-3.779310, -3.283186],
        array![3.584428, -1.848126],
    ]
}

#[test]
fn himmelblau_integration() {
    run_integration_test(&himmelblau_ideal(), 0.1, |command| {
        command
            .arg("run")
            .arg("--seed=25565")
            .arg("--max-nevals=70")
            .arg("--initial=20")
            .arg("--popsize=8")
            .arg("--param=x1 real -5 5")
            .arg("--param=x2 real -5 5")
            .args(&["function", "himmelblau"]);
    })
}

#[test]
fn rastrigin_d2_integration() {
    run_integration_test(&[vec![0.0; 2].into()], 0.2, |command| {
        command
            .arg("run")
            .arg("--seed=17418")
            .arg("--max-nevals=90")
            .arg("--initial=30")
            .arg("--popsize=5")
            .arg("--param=x1 real -5.12 5.12")
            .arg("--param=x2 real -5.12 5.12")
            .args(&["function", "rastrigin"]);
    })
}

/// Finding the optimum for many dimensions is really difficult…
#[test]
fn rastrigin_d5_integration() {
    run_integration_test(&[vec![0.0; 5].into()], 3.0, |command| {
        command
            .arg("run")
            .arg("--seed=10763")
            .arg("--max-nevals=110")
            .arg("--initial=60")
            .arg("--popsize=5")
            .arg("--param=x1 real -5.12 5.12")
            .arg("--param=x2 real -5.12 5.12")
            .arg("--param=x3 real -5.12 5.12")
            .arg("--param=x4 real -5.12 5.12")
            .arg("--param=x5 real -5.12 5.12")
            .args(&["function", "rastrigin"]);
    })
}

#[test]
fn rosenbrock_d2_integration() {
    run_integration_test(&[vec![1.0; 2].into()], 0.1, |command| {
        command
            .arg("run")
            .arg("--seed=19748")
            .arg("--max-nevals=60")
            .arg("--initial=25")
            .arg("--popsize=7")
            .arg("--param=x1 real -2.5 2.5")
            .arg("--param=x2 real -2.5 2.5")
            .args(&["function", "rosenbrock"]);
    })
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

    let stats = result.suggestion_statistics();
    let value = stats.mean();
    let value_std = stats.std();

    let guess: Array1<f64> = result
        .suggestion_location()
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
