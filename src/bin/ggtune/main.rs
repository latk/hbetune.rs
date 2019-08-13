extern crate ggtune;
extern crate strfmt;
#[macro_use]
extern crate structopt;
extern crate itertools;
extern crate ndarray;
#[macro_use]
extern crate failure;

use structopt::StructOpt as _;

mod objective_shell;

#[derive(Debug, StructOpt)]
struct CliApp {
    #[structopt(subcommand)]
    command: CliCommand,
}

#[derive(Debug, StructOpt)]
enum CliCommand {
    #[structopt(name = "run")]
    Run(CliCommandRun),
}

#[derive(Debug, StructOpt)]
struct CliCommandRun {
    /// Dimensions for the space. Should have form '<name> real <lo> <hi>'.
    #[structopt(long, raw(min_values = "1", number_of_values = "1"))]
    param: Vec<ggtune::Parameter>,

    /// Random number generator seed for reproducible runs.
    #[structopt(long, default_value = "7861")]
    seed: usize,

    #[structopt(flatten)]
    minimizer: ggtune::Minimizer,

    #[structopt(subcommand)]
    objective: CliObjective,
}

#[derive(Debug, StructOpt)]
enum CliObjective {
    /// As the objective function, execute an external program.
    #[structopt(name = "command")]
    Command {
        /// The shell command to invoke for each sample.
        /// Can substitute parameter values by name.
        /// E.g. `./objective "{x1}" --param={x2}`
        #[structopt(name = "objective-command", raw(min_values = "1"))]
        objective_command: Vec<String>,
    },

    /// As the objective function, use a built-in benchmark function.
    #[structopt(name = "function")]
    Function(CliBenchFunction),
}

#[derive(Debug, StructOpt)]
struct CliBenchFunction {
    /// Standard deviation of test function noise.
    #[structopt(long, default_value = "0.0")]
    noise: f64,

    /// Name of the function.
    /// (sphere, goldstein-price, easom, himmelblau, rastrigin, rosenbrock, onemax)
    function: BenchFn,
}

impl CliObjective {
    fn into_objective<'a>(
        self,
        space: &ggtune::Space,
    ) -> Box<dyn ggtune::ObjectiveFunction<f64> + 'a> {
        match self {
            CliObjective::Command { objective_command } => Box::new(
                objective_shell::RunCommandAsObjective::new(objective_command, space.clone()),
            ),
            CliObjective::Function(f) => Box::new(f),
        }
    }
}

#[derive(Debug)]
enum BenchFn {
    Sphere,
    GoldsteinPrice,
    Easom { amplitude: f64 },
    Himmelblau,
    Rastrigin { amplitude: f64 },
    Rosenbrock,
    Onemax,
}

impl std::str::FromStr for BenchFn {
    type Err = failure::Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name.to_ascii_lowercase().as_ref() {
            "sphere" => BenchFn::Sphere,
            "goldstein-price" => BenchFn::GoldsteinPrice,
            "easom" => BenchFn::Easom { amplitude: 1.0 },
            "himmelblau" => BenchFn::Himmelblau,
            "rastrigin" => BenchFn::Rastrigin { amplitude: 10.0 },
            "rosenbrock" => BenchFn::Rosenbrock,
            "onemax" | "sum" => BenchFn::Onemax,
            _ => bail!("expected sphere, got: {:?}", name),
        })
    }
}

impl ggtune::ObjectiveFunction<f64> for CliBenchFunction {
    fn run(&self, xs: ndarray::ArrayView1<f64>, rng: &mut ggtune::RNG) -> (f64, f64) {
        use ggtune::benchfn;
        match self.function {
            BenchFn::GoldsteinPrice | BenchFn::Easom { .. } | BenchFn::Himmelblau => assert_eq!(
                xs.len(),
                2,
                "objective function requires exactly two dimensions"
            ),
            BenchFn::Sphere | BenchFn::Rastrigin { .. } | BenchFn::Rosenbrock | BenchFn::Onemax => {
            }
        };
        let y = match self.function {
            BenchFn::Sphere => benchfn::sphere(xs),
            BenchFn::GoldsteinPrice => benchfn::goldstein_price(xs[0], xs[1]),
            BenchFn::Easom { amplitude } => benchfn::easom(xs[0], xs[1], amplitude),
            BenchFn::Himmelblau => benchfn::himmelblau(xs[0], xs[1]),
            BenchFn::Rastrigin { amplitude } => benchfn::rastrigin(xs, amplitude),
            BenchFn::Rosenbrock => benchfn::rosenbrock(xs),
            BenchFn::Onemax => benchfn::onemax(xs),
        };
        (rng.normal(y, self.noise), Default::default())
    }
}

fn main() {
    let args = CliApp::from_args();
    println!("args: {:#?}", args);
    match args.command {
        CliCommand::Run(run) => command_run(run),
    }
}

fn command_run(cfg: CliCommandRun) {
    use ggtune::{EstimatorGPR, MinimizerArgs, ObjectiveFunction};
    let CliCommandRun {
        param: params,
        seed,
        minimizer,
        objective,
    } = cfg;

    let mut space = ggtune::Space::new();
    for param in params {
        space.add_parameter(param.clone());
    }

    let mut rng = ggtune::RNG::new_with_seed(seed);

    let objective: Box<dyn ObjectiveFunction<f64>> = objective.into_objective(&space);

    let mut args = MinimizerArgs::<f64, EstimatorGPR>::default();
    args.output
        .add_human_readable_individuals(std::io::stdout(), &space);

    let result = minimizer
        .minimize(objective.as_ref(), space, &mut rng, args)
        .expect("minimization should proceed successfully");
    println!(
        "best sample: {:?}",
        result
            .best_individual()
            .expect("best individual should exist"),
    );
}