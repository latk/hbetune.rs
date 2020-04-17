extern crate hbetune;
extern crate ndarray;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate approx;

use hbetune::{Estimator as _, EstimatorGPR, Space, SurrogateModel as _, SurrogateModelGPR, RNG};
use ndarray::prelude::*;

struct SimpleModel {
    space: Space,
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
        let x = self.space.project_into_features(&[x.into()]);
        self.model.predict_mean(x.into())
    }

    fn uncertainty(&self, x: f64) -> f64 {
        let x = self.space.project_into_features(&[x.into()]);
        self.model.predict_statistics(x.into()).std()
    }
}

macro_rules! assert_is_close {
    ($left:expr, $right:expr, epsilon $epsilon:expr) => {
        match ($left, $right, $epsilon) {
            (left, right, epsilon) => assert!(
                abs_diff_eq!(left, right, epsilon = epsilon),
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

mod describe_gpr {
    use super::*;

    fn make_space() -> Space {
        let mut space = Space::new();
        space.add_real_parameter("test", 0.0, 1.0);
        space
    }

    mod with_differing_sampling_density {
        use super::*;

        fn make_model() -> SimpleModel {
            let xs = array![0.1, 0.5, 0.5, 0.9].insert_axis(Axis(1));
            let ys = array![1.0, 1.8, 2.2, 3.0];
            let space = make_space();
            let model = <hbetune::EstimatorGPR as hbetune::Estimator<f64>>::new(&space)
                .estimate(xs, ys, None, &mut RNG::new_with_seed(123))
                .unwrap();
            SimpleModel { space, model }
        }

        #[test]
        fn should_roughly_fit_the_data() {
            let model = make_model();
            let xs = array![0.1, 0.5, 0.9];
            let expected_ys = array![1.0, 2.0, 3.0];
            let predicted_ys = xs.mapv(|x| model.predict(x));
            let predicted_std = xs.mapv(|x| model.uncertainty(x));
            eprintln!("us = {} std = {}", predicted_ys, predicted_std);
            assert_abs_diff_eq!(predicted_ys, expected_ys, epsilon = 0.1);
            assert_is_close!(predicted_ys, expected_ys, epsilon 0.1);
        }

        #[test]
        fn should_provie_a_reasonable_interpolation() {
            let model = make_model();
            assert_is_close!(model.predict(0.3), 1.5, epsilon 0.1);
            assert_is_close!(model.predict(0.7), 2.5, epsilon 0.1);
        }

        #[test]
        fn should_prefer_a_conservative_extrapolation() {
            let model = make_model();
            assert_is_close!(model.predict(0.0), 0.9, epsilon 0.1);
            assert_is_close!(model.predict(1.0), 3.1, epsilon 0.1);
        }

        #[test]
        fn should_have_similar_uncertainty_for_single_observations() {
            let model = make_model();
            assert_is_close!(model.uncertainty(0.1), model.uncertainty(0.9), epsilon 0.05);
        }

        #[test]
        fn should_have_lower_uncertainty_for_more_observations() {
            let model = make_model();
            assert_relation!(operator <, model.uncertainty(0.5), model.uncertainty(0.1));
        }
    }

    mod with_unsampled_regions {
        use super::*;

        fn make_model() -> SimpleModel {
            let xs = array![0.3, 0.5, 0.7].insert_axis(Axis(1));
            let ys = array![1.0, 2.0, 1.5];
            let space = make_space();
            let model = <hbetune::EstimatorGPR as hbetune::Estimator<f64>>::new(&space)
                .noise_bounds(1e-5, 1e0)
                .length_scale_bounds(vec![(0.1, 1.0)])
                .estimate(xs, ys, None, &mut RNG::new_with_seed(9372))
                .unwrap();
            eprintln!("estimated mode: {:#?}", model);
            SimpleModel { space, model }
        }

        #[test]
        fn has_low_uncertainty_at_samples() {
            let model = make_model();
            assert_relation!(operator <, model.uncertainty(0.3), 0.01);
            assert_relation!(operator <, model.uncertainty(0.5), 0.01);
            assert_relation!(operator <, model.uncertainty(0.7), 0.01);
        }

        #[test]
        fn should_have_more_uncertainty_for_interpolation() {
            let model = make_model();
            assert_relation!(operator >, model.uncertainty(0.4), 10. * model.uncertainty(0.3));
            assert_relation!(operator >, model.uncertainty(0.6), 10. * model.uncertainty(0.3));
        }

        #[test]
        fn should_have_more_uncertainty_for_extrapolation() {
            let model = make_model();
            assert_relation!(operator >, model.uncertainty(0.0), 10. * model.uncertainty(0.3));
            assert_relation!(operator >, model.uncertainty(1.0), 10. * model.uncertainty(0.3));
        }
    }

    #[test]
    fn works_in_1d() {
        use hbetune::benchfn::sphere;

        let xs = Array::linspace(-2.0, 2.0, 5).into_shape((5, 1)).unwrap();
        let ys: Array1<_> = xs.outer_iter().map(sphere).collect();
        assert_is_close!(ys.view(), array![4.0, 1.0, 0.0, 1.0, 4.0], epsilon 1e-5);

        let mut space = Space::new();
        space.add_real_parameter("x1", -2.0, 2.0);
        let model = <hbetune::EstimatorGPR as hbetune::Estimator<f64>>::new(&space)
            .length_scale_bounds(vec![(1e-2, 1e1)])
            .noise_bounds(1e-2, 1e1)
            .estimate(xs.clone(), ys, None, &mut RNG::new_with_seed(4531))
            .unwrap();
        eprintln!("trained model: {:#?}", model);

        let check_predictions = |xs: &Array2<f64>| {
            let expected_ys: Array1<_> = xs.outer_iter().map(sphere).collect();

            let (predicted_ys, predicted_std): (Vec<f64>, Vec<f64>) = xs
                .outer_iter()
                .map(|x| {
                    let stats = model.predict_statistics(x.to_owned());
                    (stats.mean(), stats.std())
                })
                .unzip();

            let is_ok = izip!(
                expected_ys.outer_iter(),
                predicted_ys.iter(),
                predicted_std.iter()
            )
            .all(|(expected, &y, &std)| {
                let expected = expected.to_owned().into_scalar();
                expected - 0.6 * std < y && y < expected + std
            });

            assert!(
                is_ok,
                "expected values were not within the predicted 1σ region\n\
                 *   expected ys: {}\n\
                 *  predicted ys: {}\n\
                 * predicted std: {}\n",
                expected_ys,
                Array1::from(predicted_ys),
                Array1::from(predicted_std)
            );
        };

        check_predictions(&xs);
        check_predictions(&array![[-1.5], [-0.5], [1.5]]);
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

mod describe_2d {
    use super::*;

    parametrize!(it_works(
        { gridtraining: TrainingSet::Grid, randomtraining: TrainingSet::Random },
        { nonoise: 0.0, lownoise: 0.1, mediumnoise: 1.0, highnoise: 4.0 },
        { selftest: TestMode::SelfTest, newsample: TestMode::NewSample },
    ));

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum TestMode {
        SelfTest,
        NewSample,
    }

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum TrainingSet {
        Grid,
        Random,
    }

    struct Configuration {
        training: TrainingSet,
        testing: TestMode,
        noise_level: f64,
        seed: usize,
    }

    struct Results {
        test_x1: Vec<f64>,
        test_x2: Vec<f64>,
        expected_y: Vec<f64>,
        actual_y: Vec<f64>,
        actual_std: Vec<f64>,
    }

    struct Record {
        test_x1: f64,
        test_x2: f64,
        expected_y: f64,
        actual_y: f64,
        actual_std: f64,
    }

    fn it_works(training_set: TrainingSet, noise_level: f64, testmode: TestMode) {
        let seeds = [1234, 171718, 6657, 8877, 4184, 8736, 2712, 12808];

        let mut errors: Vec<_> = seeds
            .iter()
            .cloned()
            .map(|seed| {
                let conf = Configuration {
                    training: training_set,
                    testing: testmode,
                    noise_level,
                    seed,
                };

                let results = conf.run();
                results.look_good(&conf).err()
            })
            .flatten()
            .collect();

        // skipping a bad seed is acceptable
        if errors.len() > 1 {
            panic!(errors.pop().unwrap()())
        }
    }

    impl Configuration {
        fn allowed_noise(&self) -> f64 {
            let Configuration {
                training,
                testing,
                noise_level,
                seed: _seed,
            } = *self;

            let cond = |cond, value| if cond { Some(value) } else { None };

            [
                Some(noise_level),
                Some(0.1),
                cond(training == TrainingSet::Random, 0.1),
                cond(testing == TestMode::NewSample, 0.1),
                cond(testing == TestMode::SelfTest && noise_level == 0.0, 0.1),
            ]
            .iter()
            .flatten()
            .sum()
        }

        fn allowed_failures(&self) -> usize {
            let Configuration {
                training,
                testing,
                noise_level,
                seed: _seed,
            } = *self;

            (noise_level > 1.0) as usize
                + (testing == TestMode::NewSample) as usize
                + ((training, testing) == (TrainingSet::Random, TestMode::NewSample)) as usize
        }

        fn run(&self) -> Results {
            use hbetune::benchfn::sphere;

            let Configuration {
                training,
                testing,
                noise_level,
                seed,
            } = *self;

            let mut rng = RNG::new_with_seed(seed);

            let xs_features = generate_training_set(training, &mut rng);

            let ys = xs_features
                .map_axis(Axis(1), sphere)
                .mapv_into(|y| rng.normal(y, noise_level));

            let model = train_model(xs_features.clone(), ys, &mut rng);

            let xs_test_features = testing.generate_test_set(xs_features.view(), &mut rng);

            let expected_ys = xs_test_features.map_axis(Axis(1), sphere);
            let (predicted_ys, predicted_std): (Vec<_>, Vec<_>) = xs_test_features
                .outer_iter()
                .map(|x| {
                    let stats = model.predict_statistics(x.to_owned());
                    (stats.mean(), stats.std())
                })
                .unzip();

            Results {
                test_x1: xs_test_features.index_axis(Axis(1), 0).to_vec(),
                test_x2: xs_test_features.index_axis(Axis(1), 1).to_vec(),
                expected_y: expected_ys.to_vec(),
                actual_y: predicted_ys,
                actual_std: predicted_std,
            }
        }
    }

    fn generate_training_set(training_set: TrainingSet, rng: &mut RNG) -> Array2<f64> {
        match training_set {
            TrainingSet::Random => Array::zeros((50, 2)).mapv_into(|_| rng.uniform(-2.0..=2.0)),
            TrainingSet::Grid => {
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
        }
    }

    impl TestMode {
        fn generate_test_set(self, training_set: ArrayView2<f64>, rng: &mut RNG) -> Array2<f64> {
            match self {
                Self::SelfTest => training_set.to_owned(),
                Self::NewSample => Array::from_shape_fn((25, 2), |(i, _)| {
                    if i < 15 {
                        rng.uniform(-2.0..=2.0)
                    } else {
                        rng.uniform(-1.0..=1.0)
                    }
                }),
            }
        }
    }

    fn train_model(xs: Array2<f64>, ys: Array1<f64>, rng: &mut RNG) -> SurrogateModelGPR<f64> {
        let mut space = Space::new();
        space.add_real_parameter("x", -2.0, 2.0);
        space.add_real_parameter("y", -2.0, 2.0);

        <EstimatorGPR as hbetune::Estimator<f64>>::new(&space)
            .length_scale_bounds(vec![(1e-2, 2e1); 2])
            .noise_bounds(1e-2, 1e1)
            .n_restarts_optimizer(1)
            .estimate(xs.clone(), ys, None, rng)
            .expect("model should train successfully")
    }

    struct RecordIterator<'a> {
        results: &'a Results,
        i: usize,
    }

    impl<'a> Iterator for RecordIterator<'a> {
        type Item = Record;

        fn next(&mut self) -> Option<Self::Item> {
            let Self {
                ref results,
                ref mut i,
            } = self;
            if *i < results.len() {
                let item = Record {
                    test_x1: results.test_x1[*i],
                    test_x2: results.test_x2[*i],
                    expected_y: results.expected_y[*i],
                    actual_y: results.actual_y[*i],
                    actual_std: results.actual_std[*i],
                };
                *i += 1;
                Some(item)
            } else {
                None
            }
        }
    }

    #[allow(dead_code)]
    enum Never {}

    impl Results {
        fn len(&self) -> usize {
            self.test_x1.len()
        }

        fn records(&self) -> RecordIterator {
            RecordIterator {
                results: self,
                i: 0,
            }
        }

        fn global_mse(&self) -> f64 {
            let sum: f64 = self
                .actual_y
                .iter()
                .zip(&self.expected_y)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum();
            let mean = sum / (self.len() as f64);
            mean.sqrt()
        }

        fn look_good(self, conf: &Configuration) -> Result<(), Box<dyn FnOnce() -> Never>> {
            let allowed_noise = conf.allowed_noise();
            let allowed_failures = conf.allowed_failures();
            let seed = conf.seed;

            let global_mse = self.global_mse();
            if global_mse > allowed_noise {
                return Err(Box::new(move || {
                    panic!(
                        "Too large average error!\n\
                         seed: {}\n\
                         global_mse:    {}\n\
                         allowed_noise: {}\n\
                         --- samples:\n\
                         {}\n\
                         ---",
                        seed, global_mse, allowed_noise, self,
                    )
                }));
            }

            let prediction_not_ok: Results = self
                .records()
                .filter(|r| !r.y_is_within_plausible_bounds(-2.0, 1.0, allowed_noise))
                .collect();
            let prediction_not_ok_count = prediction_not_ok.len();
            if prediction_not_ok_count > allowed_failures {
                return Err(Box::new(move || {
                    panic!(
                        "Incorrect prediction ({})\n\
                         seed: {}\n\
                         --- samples:\n\
                         {}\n\
                         ---",
                        seed, prediction_not_ok_count, prediction_not_ok,
                    )
                }));
            }

            let std_not_ok: Results = self
                .records()
                .filter(|r| !r.std_is_within_reasonable_bounds(1.5 * allowed_noise))
                .collect();
            let std_not_ok_count = std_not_ok.len();
            if std_not_ok_count > allowed_failures {
                return Err(Box::new(move || {
                    panic!(
                        "Detected large variances ({} over {})\n\
                         seed: {}\n\
                         ---\n\
                         {}\n\
                         ---",
                        seed,
                        std_not_ok_count,
                        1.5 * allowed_noise,
                        std_not_ok,
                    )
                }));
            }

            Ok(())
        }
    }

    impl std::fmt::Display for Results {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let mut want_newline = false;
            for r in self.records() {
                if std::mem::replace(&mut want_newline, true) {
                    writeln!(f)?;
                }
                write!(
                    f,
                    "x1={:+1.10} x2={:+1.10} y={:+1.10} | y={:+1.10} std={:+1.10}",
                    r.test_x1, r.test_x2, r.expected_y, r.actual_y, r.actual_std,
                )?;
            }
            Ok(())
        }
    }

    impl std::iter::FromIterator<Record> for Results {
        fn from_iter<I>(records: I) -> Self
        where
            I: IntoIterator<Item = Record>,
        {
            let records = Vec::from_iter(records);
            Results {
                test_x1: records.iter().map(|r| r.test_x1).collect(),
                test_x2: records.iter().map(|r| r.test_x2).collect(),
                expected_y: records.iter().map(|r| r.expected_y).collect(),
                actual_y: records.iter().map(|r| r.actual_y).collect(),
                actual_std: records.iter().map(|r| r.actual_std).collect(),
            }
        }
    }

    impl Record {
        fn y_is_within_plausible_bounds(
            &self,
            zscore_lo: f64,
            zscore_hi: f64,
            allowed_noise: f64,
        ) -> bool {
            let Self {
                expected_y,
                actual_y,
                actual_std,
                ..
            } = *self;
            let bound_lo = expected_y + zscore_lo * actual_std + zscore_lo.signum() * allowed_noise;
            let bound_hi = expected_y + zscore_hi * actual_std + zscore_hi.signum() * allowed_noise;
            bound_lo <= actual_y && actual_y <= bound_hi
        }

        fn std_is_within_reasonable_bounds(&self, bound: f64) -> bool {
            let Self { actual_std, .. } = *self;
            actual_std <= bound
        }
    }
}
