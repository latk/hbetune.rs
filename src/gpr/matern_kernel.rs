use ndarray::prelude::*;
use num_traits::Float;

use crate::gpr::Kernel;
use crate::gpr::Scalar;
use crate::util::{BoundedValue, BoundsError};

/// Matern kernel family, parameterized by smoothness parameter nu.
/// Allows separate length scales so that it can be used as an anisotropic kernel.
/// Nu should be of for p/2, e.g. 3/2.
#[derive(Clone, Debug)]
pub struct Matern {
    nu: f64,
    length_scale: Vec<BoundedValue<f64>>,
}

impl Matern {
    /// Create a new matern kernel with certain smoothness and length scales.
    pub fn new(nu: f64, length_scale: Vec<BoundedValue<f64>>) -> Self {
        Matern { nu, length_scale }
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    pub fn length_scale(&self) -> &[BoundedValue<f64>] {
        self.length_scale.as_slice()
    }

    fn length_scale_array(&self) -> Array1<f64> {
        self.length_scale.iter().map(BoundedValue::value).collect()
    }
}

impl Kernel for Matern {
    fn kernel<A: Scalar>(&self, x1: ArrayView2<A>, x2: ArrayView2<A>) -> Array2<A> {
        assert_eq!(
            x1.cols(),
            self.n_params(),
            "number of x1 columns must match number of features",
        );
        assert_eq!(
            x2.cols(),
            self.n_params(),
            "number of x2 columns must match number of features",
        );

        // normalize the arrays via their length scale
        let length_scale = self.length_scale_array().mapv(A::from_f);
        let normalize_length_scale = |xs: ArrayView2<A>| {
            xs.to_owned()
                / length_scale
                    .view()
                    .insert_axis(Axis(0))
                    .broadcast(xs.shape())
                    .unwrap()
        };
        let x1 = normalize_length_scale(x1);
        let x2 = normalize_length_scale(x2);

        // raw euclidean distance matrix
        let dists = cdist::cdist(x1.view(), x2.view());

        match self.nu {
            nu if approx_eq!(f64, nu, 0.5) => (-dists).mapv(Float::exp),
            nu if approx_eq!(f64, nu, 1.5) => {
                let kernel: Array2<A> = dists * A::from_f(3.0.sqrt());
                (kernel.clone() + A::from_i(1)) * (-kernel).mapv(Float::exp)
            }
            nu if approx_eq!(f64, nu, 2.5) => {
                // K = dists * sqrt(4)
                let kernel: Array2<A> = dists * A::from_f(5.0.sqrt());
                // (1 + K + K**2 / 3) * exp(-K)
                (kernel.mapv(|x| x.powi(2)) / A::from_i(3) + &kernel + A::from_i(1))
                    * (-kernel).mapv(Float::exp)
            }
            _ => unimplemented!("Matern kernel with arbitrary values for nu"),
        }
    }

    fn theta_grad<A: Scalar>(&self, x: ArrayView2<A>) -> (Array2<A>, Array3<A>) {
        let kernel = self.kernel(x, x);
        let gradient_shape = (x.rows(), x.rows(), self.n_params());
        let length_scale = self.length_scale_array().mapv(A::from_f);

        // pairwise dimension-wise distances: d[ijk] = (x[ik] - x[jk])**2 / length_scale[k]**2
        let d_shape = (x.rows(), x.rows(), x.cols());
        let x_ik = x.insert_axis(Axis(1));
        let x_ik = x_ik.broadcast(d_shape).unwrap();
        let x_jk = x.insert_axis(Axis(0));
        let x_jk = x_jk.broadcast(d_shape).unwrap();
        let scales_k_square = length_scale
            .mapv(|x| x.powi(2))
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));
        let d = (&x_ik - &x_jk).mapv(|x| x.powi(2)) / scales_k_square;
        assert_eq!(d.shape(), &[x.rows(), x.rows(), self.n_params()]);

        let gradient = match self.nu {
            nu if approx_eq!(f64, nu, 0.5) => {
                let mut gradient = kernel.clone().insert_axis(Axis(2)) * &d
                    / d.sum_axis(Axis(2)).mapv(Float::sqrt).insert_axis(Axis(2));
                gradient.map_inplace(|x| {
                    if !x.is_finite() {
                        *x = 0f32.into();
                    }
                });
                gradient
            }
            nu if approx_eq!(f64, nu, 1.5) => {
                // gradient = 3 * d * exp(-sqrt(3 * d.sum(-1)))[..., np.newaxis]
                let tmp =
                    (-(d.sum_axis(Axis(2)) * A::from_i(3)).mapv(Float::sqrt)).mapv(Float::exp);
                &d.broadcast(gradient_shape).unwrap()
                    * &tmp.insert_axis(Axis(2)).broadcast(gradient_shape).unwrap()
                    * A::from_i(3)
            }
            nu if approx_eq!(f64, nu, 2.5) => {
                let tmp: Array3<A> = (d.sum_axis(Axis(2)) * A::from_i(5))
                    .mapv(Float::sqrt)
                    .insert_axis(Axis(2));
                (-tmp.clone())
                    .mapv(Float::exp)
                    .broadcast(gradient_shape)
                    .unwrap()
                    .to_owned()
                    * (tmp + A::from_i(1)).broadcast(gradient_shape).unwrap()
                    * d.broadcast(gradient_shape).unwrap()
                    * A::from_f(5. / 3.)
            }
            _ => unimplemented!("Matern kernel gradient with arbitrary values for nu"),
        };
        (kernel, gradient)
    }

    fn diag<A: Scalar>(&self, x: ArrayView2<A>) -> Array1<A> {
        Array::ones(x.rows())
    }

    fn n_params(&self) -> usize {
        self.length_scale.len()
    }

    fn theta(&self) -> Vec<f64> {
        self.length_scale
            .iter()
            .map(BoundedValue::value)
            .map(f64::ln)
            .collect()
    }

    fn with_theta(self, theta: &[f64]) -> Result<Self, BoundsError<f64>> {
        assert!(theta.len() == self.n_params());

        let length_scale = theta
            .iter()
            .cloned()
            .map(f64::exp)
            .zip(self.length_scale)
            .map(|(value, bounded)| bounded.with_value(value))
            .collect::<Result<_, _>>()?;

        Ok(Self::new(self.nu, length_scale))
    }

    fn with_clamped_theta(self, theta: &[f64]) -> Self {
        assert!(theta.len() == self.n_params());

        let length_scale = theta
            .iter()
            .cloned()
            .map(f64::exp)
            .zip(self.length_scale)
            .map(|(value, bounded)| bounded.with_clamped_value(value))
            .collect::<Vec<_>>();

        Self::new(self.nu, length_scale)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.length_scale
            .iter()
            .map(|bounds| (bounds.min().ln(), bounds.max().ln()))
            .collect()
    }
}

#[test]
#[allow(clippy::unreadable_literal)]
fn it_produces_a_kernel_and_gradient_with_nu_3_2() {
    let kernel = Matern::new(
        1.5,
        vec![
            BoundedValue::new(1., 0.05, 20.).unwrap(),
            BoundedValue::new(1., 0.05, 20.).unwrap(),
        ],
    );

    let x = array![[0., 0.], [1., 1.], [1., 2.]];

    let kernel_matrix = array![
        // produced by sklearn
        [1., 0.29782077, 0.1013397],
        [0.29782077, 1., 0.48335772],
        [0.1013397, 0.48335772, 1.]
    ];

    let gradient_matrix = array![
        // produced by sklearn
        [[0., 0.], [0.25901289, 0.25901289], [0.0623887, 0.24955481]],
        [[0.25901289, 0.25901289], [0., 0.], [0., 0.53076362]],
        [[0.0623887, 0.24955481], [0., 0.53076362], [0., 0.]]
    ];

    let (actual_kernel, actual_gradient) = kernel.theta_grad(x.view());
    assert_all_close!(&actual_kernel, &kernel_matrix, 0.001);
    assert_all_close!(actual_gradient, gradient_matrix, 0.001);
    assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
}

#[test]
#[allow(clippy::unreadable_literal)]
fn it_produces_a_kernel_and_gradient_with_nu_5_2() {
    let kernel = Matern::new(
        2.5,
        vec![
            BoundedValue::new(1., 0.05, 20.).unwrap(),
            BoundedValue::new(1., 0.05, 20.).unwrap(),
        ],
    );

    let x = array![[0., 0.], [1., 1.], [1., 2.]];

    let kernel_matrix = array![
        // produced by sklearn
        [1., 0.31728336, 0.09657724],
        [0.31728336, 1., 0.52399411],
        [0.09657724, 0.52399411, 1.]
    ];

    let gradient_matrix = array![
        // produced by sklearn
        [[0., 0.], [0.29364328, 0.29364328], [0.06737947, 0.26951788]],
        [[0.29364328, 0.29364328], [0., 0.], [0., 0.57644039]],
        [[0.06737947, 0.26951788], [0., 0.57644039], [0., 0.]]
    ];

    let (actual_kernel, actual_gradient) = kernel.theta_grad(x.view());
    assert_all_close!(&actual_kernel, &kernel_matrix, 0.001);
    assert_all_close!(actual_gradient, gradient_matrix, 0.001);
    assert_all_close!(kernel.diag(x.view()), actual_kernel.diag(), 1e-3);
}

mod cdist {
    use crate::gpr::Scalar;
    use ndarray::prelude::*;

    #[cfg(test)]
    use num_traits::Float;

    pub fn cdist<A: Scalar>(xa: ArrayView2<A>, xb: ArrayView2<A>) -> Array2<A> {
        let mut out = Array2::zeros((xa.rows(), xb.rows()));
        for (i, a) in xa.outer_iter().enumerate() {
            for (j, b) in xb.outer_iter().enumerate() {
                out[[i, j]] = (&a - &b).mapv_into(|x| x.powi(2)).sum().sqrt();
            }
        }
        out
    }

    #[test]
    fn it_calculates_a_single_distance() {
        let a = array![[1., 3.]];
        let b = array![[2., 5.]];
        // (1 - 2)**2 + (3 - 5)**2 = 1**2 + 2**2 = 5
        assert_eq!(cdist(a.view(), b.view()), array![[5.0.sqrt()]]);
    }

    #[test]
    fn it_calculates_multiple_distances() {
        let a = array![[0., 0.], [1., 1.], [2., 2.]];
        let b = array![[1., 2.], [3., 4.]];
        assert_eq!(
            cdist(a.view(), b.view()),
            array![
                [5.0.sqrt(), 25.0.sqrt()],
                [1.0.sqrt(), 13.0.sqrt()],
                [1.0.sqrt(), 5.0.sqrt()]
            ]
        );
    }
}
