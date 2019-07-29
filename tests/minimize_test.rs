extern crate ggtune;
#[macro_use] extern crate ndarray;
extern crate itertools;
use itertools::Itertools as _;

#[test]
fn minimize_works() {
    type Minimizer = ggtune::Minimizer<f64, ggtune::EstimatorGPR>;

    let mut minimizer = Minimizer::default();
    minimizer.max_nevals = 40;

    let mut space = ggtune::Space::new();
    space.add_real_parameter("x1", -2.0, 3.0);
    space.add_real_parameter("x2", -3.0, 2.0);

    let mut rng = ggtune::RNG::new_with_seed(904827189);

    fn objective(xs: ndarray::ArrayView1<f64>, _rng: &mut ggtune::RNG) -> (f64, f64) {
        (ggtune::benchfn::sphere(xs), 0.0)
    }

    let result = minimizer.minimize(Box::new(objective), space, &mut rng, None, vec![])
        .expect("minimization should proceed successfully");

    let guess = result.best_individual()
        .expect("there should be a best individual")
        .sample();
    let optimizer = array![0.0, 0.0];
    let distance = (&optimizer - &guess).mapv(|x| x.powi(2)).sum().sqrt();
    assert!(distance < 0.2,
            "optimal point was not found\n\
             |  distance: {}\n\
             | optimizer: {}\n\
             |     guess: {}\n\
             |        ys: {:.2}",
            distance, optimizer, guess, result.ys().unwrap().iter().format(", "));
}
