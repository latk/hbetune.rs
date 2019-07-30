#[macro_use]
extern crate ggtune;
extern crate speculate;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate itertools;

use ggtune::{Estimator as _, EstimatorGPR, Space, SurrogateModel as _, SurrogateModelGPR, RNG};
use itertools::Itertools as _;
use ndarray::prelude::*;
use speculate::speculate;

struct SimpleModel {
    model: SurrogateModelGPR<f64>,
}

impl std::fmt::Debug for SimpleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("SimpleModel")
            .field(self.model.kernel())
            .finish()
    }
}

impl SimpleModel {
    fn predict(&self, x: f64) -> f64 {
        self.model.predict_mean(array![x])
    }

    fn uncertainty(&self, x: f64) -> f64 {
        self.model.predict_mean_std(array![x]).1
    }
}

macro_rules! assert_is_close {
    ($left:expr, $right:expr, epsilon $epsilon:expr) => {
        match ($left, $right, $epsilon) {
            (left, right, epsilon) => assert!(
                (left - right).abs() < epsilon,
                "expected ({}) == ({}) with epsilon {} but found\n\
                 |  left: {:?}\n\
                 | right: {:?}",
                stringify!($left),
                stringify!($right),
                epsilon,
                left,
                right
            ),
        }
    };
}

macro_rules! assert_relation {
    (operator $operator:tt, $left:expr, $right:expr) => {
        match ($left, $right) {
            (left, right) => assert!(
                left $operator right,
                "expected ({}) {} ({}) but found\n\
                 |  left: {:?}\n\
                 | right: {:?}",
                stringify!($left),
                stringify!($operator),
                stringify!($right),
                left,
                right
            )
        }
    };
}

speculate! {
    describe "GPR" {

        fn make_space() -> Space {
            let mut space = Space::new();
            space.add_real_parameter("test", 0.0, 1.0);
            space
        }

        describe "differing sampling density" {

            fn make_model() -> SimpleModel {
                let xs = array![0.1, 0.5, 0.5, 0.9].insert_axis(Axis(1));
                let ys = array![1.0, 1.8, 2.2, 3.0];
                let space = make_space();
                let model = <ggtune::EstimatorGPR as ggtune::Estimator<f64>>::new(&space).estimate(
                    xs, ys, space,
                    None,
                    &mut RNG::new_with_seed(123),
                ).unwrap();
                SimpleModel { model }
            }

            it "should roughly fit the data" {
                let model = make_model();
                let xs = array![0.1, 0.5, 0.9];
                let expected_ys = array![1.0, 2.0, 3.0];
                let predicted_ys = xs.mapv(|x| model.predict(x.clone()));
                let predicted_std = xs.mapv(|x| model.uncertainty(x));
                eprintln!("us = {} std = {}", predicted_ys, predicted_std);
                assert_all_close!(predicted_ys, expected_ys, 0.1);
            }

            it "should provide a reasonable interpolation" {
                let model = make_model();
                assert_is_close!(model.predict(0.3), 1.5, epsilon 0.1);
                assert_is_close!(model.predict(0.7), 2.5, epsilon 0.1);
            }

            it "should prefer a conservative extrapolation" {
                let model = make_model();
                assert_is_close!(model.predict(0.0), 0.9, epsilon 0.1);
                assert_is_close!(model.predict(1.0), 3.1, epsilon 0.1);
            }

            it "should have similar uncertainty for single observations" {
                let model = make_model();
                assert_is_close!(model.uncertainty(0.1), model.uncertainty(0.9), epsilon 0.05);
            }

            it "should have lower uncertainty for more observations" {
                let model = make_model();
                assert_relation!(operator <, model.uncertainty(0.5), model.uncertainty(0.1));
            }
        }

        describe "unsampled regions" {

            fn make_model() -> SimpleModel {
                let xs = array![0.3, 0.5, 0.7].insert_axis(Axis(1));
                let ys = array![1.0, 2.0, 1.5];
                let space = make_space();
                let model = <ggtune::EstimatorGPR as ggtune::Estimator<f64>>::new(&space)
                    .noise_bounds(1e-5, 1e0)
                    .length_scale_bounds(vec![(0.1, 1.0)])
                    .estimate(
                        xs, ys, space,
                        None,
                        &mut RNG::new_with_seed(9372),
                    ).unwrap();
                eprintln!("estimated mode: {:#?}", model);
                SimpleModel { model }
            }

            it "has low uncertainty at samples" {
                let model = make_model();
                assert_relation!(operator <, model.uncertainty(0.3), 0.01);
                assert_relation!(operator <, model.uncertainty(0.5), 0.01);
                assert_relation!(operator <, model.uncertainty(0.7), 0.01);
            }

            it "should have more uncertainty for interpolation" {
                let model = make_model();
                assert_relation!(operator >, model.uncertainty(0.4), 10. * model.uncertainty(0.3));
                assert_relation!(operator >, model.uncertainty(0.6), 10. * model.uncertainty(0.3));
            }

            it "should have more uncertainty for extrapolation" {
                let model = make_model();
                assert_relation!(operator >, model.uncertainty(0.0), 10. * model.uncertainty(0.3));
                assert_relation!(operator >, model.uncertainty(1.0), 10. * model.uncertainty(0.3));
            }
        }

        it "works in 1d" {
            use ggtune::benchfn::sphere;

            let xs = Array::linspace(-2.0, 2.0, 5).into_shape((5, 1)).unwrap();
            let ys: Array1<_> = xs.outer_iter().map(sphere).collect();
            assert_all_close!(ys.view(), array![4.0, 1.0, 0.0, 1.0, 4.0], 1e-5);

            let mut space = Space::new();
            space.add_real_parameter("x1", -2.0, 2.0);
            let model = <ggtune::EstimatorGPR as ggtune::Estimator<f64>>::new(&space)
                .length_scale_bounds(vec![(1e-2, 1e1)])
                .noise_bounds(1e-2, 1e1)
                .estimate(
                    xs.clone(), ys,
                    space, None, &mut RNG::new_with_seed(4531),
                ).unwrap();
            eprintln!("trained model: {:#?}", model);

            let check_predictions = |xs: &Array2<_>| {
                let expected_ys: Array1<_> = xs.outer_iter().map(sphere).collect();

                let (predicted_ys, predicted_std) = model.predict_mean_std_a(xs.to_owned());

                let is_ok = izip!(expected_ys.outer_iter(),
                                  predicted_ys.outer_iter(),
                                  predicted_std.outer_iter())
                    .all(|(expected, y, std)| {
                        let expected = expected.to_owned().into_scalar();
                        let y = y.to_owned().into_scalar();
                        let std = std.to_owned().into_scalar();
                        expected - 0.6 * std < y && y < expected + std
                    });

                assert!(is_ok, "expected values were not within the predicted 1σ region\n\
                                *   expected ys: {}\n\
                                *  predicted ys: {}\n\
                                * predicted std: {}\n",
                        expected_ys, predicted_ys, predicted_std);
            };

            check_predictions(&xs);
            check_predictions(&array![[-1.5], [-0.5], [1.5]]);
        }
    }
}

/// call a parameterized test function.
///
/// Example:
///
/// ``` ignore
/// parametrize(the_test_function(
///   {a1: 1, a2: 2},
///   {bx: "x", by: "y"},
/// ));
/// ```
///
/// This will result in the tests:
///
/// ``` ignore
/// the_test_function::a1::bx::is_ok: the_test_function(1, "x")
/// the_test_function::a1::by::is_ok: the_test_function(1, "y")
/// the_test_function::a2::bx::is_ok: the_test_function(2, "x")
/// the_test_function::a2::by::is_ok: the_test_function(2, "y")
/// ```
macro_rules! parametrize {
    ($target:ident ( $($choices:tt,)* )) => (
        mod $target {
            #[allow(unused_imports)] use super::*;
            parametrize!(@consify $target, [], ( $($choices,)* ));
        }
    );

    (@consify $target:ident, $cons:tt, ()) => (
        parametrize!(@reverse $target, [], $cons);
    );

    (@consify $target:ident, $cons:tt, ( $first:tt, $($rest:tt,)* )) => (
        parametrize!(@consify $target, [$first, $cons], ( $($rest,)* ));
    );

    (@reverse $target:ident, $cons:tt, []) => (
        parametrize!(@partial $target, [], $cons);
    );

    (@reverse $target:ident, $cons:tt, [$first:tt, $rest:tt]) => (
        parametrize!(@reverse $target, [$first, $cons], $rest);
    );

    (@partial $target:ident, $old:tt, []) => (
        parametrize!(@deconsify $target, (), $old);
    );

    (@partial $target:ident, $old:tt, [{ $( $name:ident : $value:expr),* $(,)? }, $rest:tt ]) => (
        $(mod $name {
            #[allow(unused_imports)] use super::*;
            parametrize!(@partial $target, [$value, $old], $rest);
        })*
    );

    (@deconsify $target:ident, ( $($params:tt),* ), []) => (
        #[test]
        fn is_ok() {
            $target($($params),*)
        }
    );

    (@deconsify $target:ident, ( $($params:tt),* ), [$first:tt, $rest:tt]) => (
        parametrize!(@deconsify $target, ( $first $(, $params)* ), $rest);
    );
}

parametrize!(it_works_in_2d(
    { rng123: 123, rng171718: 171718, rng6657: 6657 },
    { gridtraining: "gridTraining", randomtraining: "randomTraining" },
    { nonoise: 0.0, lownoise: 0.1, mediumnoise: 1.0, highnoise: 4.0 },
    { selftest: "selftest", newsample: "newsample" },
));

fn it_works_in_2d(rng_seed: usize, training_set: &str, noise_level: f64, testmode: &str) {
    use ggtune::benchfn::sphere;

    let mut rng = RNG::new_with_seed(rng_seed);

    let xs = generate_training_set(training_set, &mut rng);

    let ys = xs
        .map_axis(Axis(1), |ax| sphere(ax))
        .mapv_into(|y| rng.normal(y, noise_level));

    let model = train_model(xs.clone(), ys, &mut rng);

    fn generate_training_set(training_set: &str, rng: &mut RNG) -> Array2<f64> {
        match training_set {
            "randomTraining" => Array::zeros((50, 2)).mapv_into(|_| rng.uniform(-2.0..=2.0)),
            "gridTraining" => {
                const SIZE: usize = 7;
                let gridaxis = Array::linspace(-2.0, 2.0, SIZE);
                Array::from_shape_fn((SIZE * SIZE, 2), |(i, j)| {
                    if j % 2 == 0 {
                        gridaxis[i % SIZE]
                    } else {
                        gridaxis[i / SIZE]
                    }
                })
            }
            _ => unimplemented!("{}", training_set),
        }
    }

    fn train_model(xs: Array2<f64>, ys: Array1<f64>, rng: &mut RNG) -> SurrogateModelGPR<f64> {
        let mut space = Space::new();
        space.add_real_parameter("x", -2.0, 2.0);
        space.add_real_parameter("y", -2.0, 2.0);

        let model = <EstimatorGPR as ggtune::Estimator<f64>>::new(&space)
            .length_scale_bounds(vec![(1e-2, 2e1); 2])
            .noise_bounds(1e-2, 1e1)
            .n_restarts_optimizer(1)
            .estimate(xs.clone(), ys, space, None, rng)
            .expect("model should train successfully");

        model
    }

    let cond = |cond, value| if cond { Some(value) } else { None };

    let allowed_failures = 0
        + (noise_level > 1.0) as usize
        + (testmode == "newsample") as usize
        + ((training_set, testmode) == ("randomTraining", "newsample")) as usize;

    let allowed_noise = [
        Some(noise_level),
        Some(0.1),
        cond(training_set == "randomTraining", 0.1),
        cond(testmode == "newsample", 0.1),
        cond(testmode == "selftest" && noise_level == 0.0, 0.1),
    ]
    .iter()
    .flatten()
    .sum();

    let xs_test = match testmode {
        "selftest" => xs.clone(),
        "newsample" => Array::from_shape_fn((25, 2), |(i, _)| {
            if i < 15 {
                rng.uniform(-2.0..=2.0)
            } else {
                rng.uniform(-1.0..=1.0)
            }
        }),
        _ => unimplemented!("{}", testmode),
    };

    let expected_ys = xs_test.map_axis(Axis(1), |ax| sphere(ax));
    let (predicted_ys, predicted_std) = model.predict_mean_std_a(xs_test.clone());

    let error = (&predicted_ys - &expected_ys)
        .mapv_into(|error| error.powi(2))
        .mean_axis(Axis(0))
        .into_scalar()
        .sqrt();

    assert_relation!(operator <, error, allowed_noise);

    fn index_locs<'a, T>(
        items: impl IntoIterator<Item = T> + 'a,
        locs: &'a Vec<bool>,
    ) -> impl Iterator<Item = T> + 'a {
        items
            .into_iter()
            .zip_eq(locs.iter().cloned())
            .filter_map(|(el, ok)| if ok { Some(el) } else { None })
    }

    // assert_eq!(index_locs(vec![1 as usize, 2], &vec![false, true]).collect_vec(),
    //            vec![2 as usize]);

    struct LocInfo<'a> {
        locs: Vec<bool>,
        xs_test: ArrayView2<'a, f64>,
        predicted_ys: ArrayView1<'a, f64>,
        expected_ys: ArrayView1<'a, f64>,
        predicted_std: ArrayView1<'a, f64>,
    };

    impl<'a> std::fmt::Debug for LocInfo<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let locs = &self.locs;
            writeln!(f, "locs [{}]", locs.iter().join(", "))?;
            writeln!(
                f,
                "x1       [{}]",
                index_locs(self.xs_test.index_axis(Axis(1), 0), locs).join(", ")
            )?;
            writeln!(
                f,
                "x2       [{}]",
                index_locs(self.xs_test.index_axis(Axis(1), 1), locs).join(", ")
            )?;

            writeln!(
                f,
                "ys our   [{}]",
                index_locs(self.predicted_ys, locs).join(", ")
            )?;
            writeln!(
                f,
                "ys expec [{}]",
                index_locs(self.expected_ys, locs).join(", ")
            )?;
            writeln!(
                f,
                "std our  [{}]",
                index_locs(self.predicted_std, locs).join(", ")
            )?;
            Ok(())
        }
    }

    let bound = |zscore: f64| {
        let bounds =
            expected_ys.clone() + zscore * &predicted_std + zscore.signum() * allowed_noise;
        bounds
    };

    let prediction_not_ok = izip!(bound(-2.0).iter(), predicted_ys.iter(), bound(1.0).iter())
        .map(|(lo, y, hi)| !(lo <= y && y <= hi))
        .collect_vec();
    let prediction_not_ok_count: usize = prediction_not_ok.iter().map(|ok| *ok as usize).sum();
    assert!(
        prediction_not_ok_count <= allowed_failures,
        "Incorrect prediction ({})\n{:?}",
        prediction_not_ok_count,
        LocInfo {
            locs: prediction_not_ok,
            xs_test: xs_test.view(),
            predicted_ys: predicted_ys.view(),
            expected_ys: expected_ys.view(),
            predicted_std: predicted_std.view(),
        }
    );

    let std_not_ok = predicted_std
        .iter()
        .map(|&std| !(std < 1.5 * allowed_noise))
        .collect_vec();
    let std_not_ok_count: usize = std_not_ok.iter().map(|&ok| ok as usize).sum();
    assert!(
        std_not_ok_count <= allowed_failures,
        "Detected large variances ({} over {})\n{:?}",
        std_not_ok_count,
        allowed_noise,
        LocInfo {
            locs: std_not_ok,
            xs_test: xs_test.view(),
            predicted_ys: predicted_ys.view(),
            expected_ys: expected_ys.view(),
            predicted_std: predicted_std.view(),
        }
    );
}
