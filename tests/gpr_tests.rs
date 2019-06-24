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
                eprintln!("ys = {} std = {}", predicted_ys, predicted_std);
                assert_all_close!(predicted_ys, expected_ys, 0.1);
            }

        }

    }
}
