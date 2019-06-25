#[macro_use] extern crate ggtune;
extern crate speculate;
#[macro_use] extern crate ndarray;

use speculate::speculate;
use ndarray::prelude::*;
use ggtune::{SurrogateModel as _, SurrogateModelGPR};

struct SimpleModel { model: SurrogateModelGPR<f64> }

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
    ($left:expr, $right:expr, epsilon $epsilon:expr) => (
        match ($left, $right, $epsilon) {
            (left, right, epsilon) =>
                assert!((left - right).abs() < epsilon,
                "expected ({}) == ({}) with epsilon {} but found\n\
                  left: {:?}\n\
                 right: {:?}",
                stringify!($left), stringify!($right), epsilon, left, right)
        }
    )
}

macro_rules! assert_relation {
    (operator $operator:tt, $left:expr, $right:expr) => (
        match ($left, $right) {
            (left, right) => assert!(
                left $operator right,
            "expected ({}) {} ({}) but found\n\
              left: {:?}\n\
              right: {:?}",
            stringify!($left), stringify!($operator), stringify!($right), left, right)
        }
    )
}

speculate! {
    describe "GPR" {
        use ggtune::{Space, RNG};

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
                let config = ggtune::ConfigGPR::new(&space);
                let model = SurrogateModelGPR::estimate(
                    xs, ys, space,
                    None,
                    &mut RNG::new_with_seed(123),
                    config,
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
                let config = ggtune::ConfigGPR::new(&space)
                    .noise_bounds(1e-3, 1e0)
                    .length_scale_bounds(vec![(0.1, 1.0)]);
                let model = SurrogateModelGPR::estimate(
                    xs, ys, space,
                    None,
                    &mut RNG::new_with_seed(9372),
                    config,
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

        }

    }
}
