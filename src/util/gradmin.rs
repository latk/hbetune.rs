extern crate nlopt;

use crate::RNG;

type Bounds = (f64, f64);

pub fn minimize_by_gradient_with_restarts<F, Data>(
    objective: &F,
    x: &mut [f64],
    bounds: &[Bounds],
    data: Data,
    restarts: usize,
    rng: &mut RNG,
) -> f64
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut Data) -> f64,
    Data: Clone,
{
    let mut bestf = minimize_by_gradient(objective, x, bounds, data.clone());
    let mut bestx = x.to_vec();
    for _ in 0..restarts {
        for (x, &(lo, hi)) in x.iter_mut().zip(bounds) {
            *x = rng.uniform(lo..=hi);
        }
        let f = minimize_by_gradient(objective, x, bounds, data.clone());
        if f < bestf {
            bestf = f;
            bestx.copy_from_slice(x);
        }
    }
    x.copy_from_slice(&bestx);
    bestf
}

pub fn minimize_by_gradient<F, Data>(
    objective: F,
    x: &mut [f64],
    bounds: &[Bounds],
    data: Data,
) -> f64
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut Data) -> f64,
{
    let (bounds_lo, bounds_hi): (Vec<f64>, Vec<f64>) = bounds.iter().cloned().unzip();
    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Lbfgs,
        x.len(),
        objective,
        nlopt::Target::Minimize,
        data,
    );
    opt.set_lower_bounds(bounds_lo.as_slice()).unwrap();
    opt.set_upper_bounds(bounds_hi.as_slice()).unwrap();
    opt.set_maxeval(150).unwrap();
    // TODO check result properly
    match opt.optimize(x) {
        Ok((_, f)) => f,
        Err((_, f)) => f,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn slanted_plane(x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) -> f64 {
        if let Some(grad) = grad {
            grad.copy_from_slice(&[1.0, 1.0]);
        }
        x.iter().sum()
    }

    mod fn_minimize_by_gradient {
        use super::*;

        #[test]
        fn works_on_a_simple_problem() {
            let mut x = vec![0.0, 0.0];
            minimize_by_gradient(slanted_plane, x.as_mut_slice(), &[(-2.0, 2.0); 2], ());
            assert_eq!(x, [-2.0, -2.0]);
        }
    }

    mod fn_minimize_by_gradient_with_restarts {
        use super::*;

        #[test]
        fn works_on_a_simple_problem() {
            let mut x = vec![0.0; 2];
            let mut rng = RNG::new_with_seed(17176);
            let f = minimize_by_gradient_with_restarts(
                &slanted_plane,
                x.as_mut_slice(),
                &[(-2.0, 2.0); 2],
                (),
                3,
                &mut rng,
            );
            assert_eq!((f, x.as_slice()), (-4.0, vec![-2.0, -2.0].as_slice()));
        }
    }
}

pub fn minimize_without_gradient<F, Data>(
    objective: F,
    x: &mut [f64],
    bounds: &[Bounds],
    data: Data,
) -> f64
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut Data) -> f64,
{
    let (bounds_lo, bounds_hi): (Vec<_>, Vec<_>) = bounds.iter().cloned().unzip();
    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Bobyqa,
        x.len(),
        objective,
        nlopt::Target::Minimize,
        data,
    );
    opt.set_lower_bounds(bounds_lo.as_slice()).unwrap();
    opt.set_upper_bounds(bounds_hi.as_slice()).unwrap();
    opt.set_maxeval(150).unwrap();
    // TODO check result properly
    match opt.optimize(x) {
        Ok((_, f)) => f,
        Err((_, f)) => f,
    }
}
