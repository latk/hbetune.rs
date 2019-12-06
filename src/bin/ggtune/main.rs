extern crate ggtune;
extern crate strfmt;
#[macro_use]
extern crate structopt;
extern crate itertools;
extern crate ndarray;
#[macro_use]
extern crate failure;
extern crate serde_json;

use itertools::Itertools as _;
use serde_json::json;
use structopt::StructOpt as _;

mod objective_shell;

#[derive(Debug, StructOpt)]
struct CliApp {
    /// enable verbose output
    #[structopt(long)]
    verbose: bool,

    /// Output less information
    #[structopt(long)]
    quiet: bool,

    #[structopt(subcommand)]
    command: CliCommand,
}

#[derive(Debug, StructOpt)]
enum CliCommand {
    /// Run the ggtune minimizer.
    #[structopt(name = "run")]
    Run(CliCommandRun),

    /// Evaluate a benchmark function.
    ///
    /// This is intended for integration with external tools.
    #[structopt(name = "function")]
    Function(CliCommandFunction),
}

#[derive(Debug, StructOpt)]
struct CliCommandRun {
    /// Dimensions for the space. Should have form '<name> real <lo> <hi>'.
    #[structopt(long, min_values = 1, number_of_values = 1)]
    param: Vec<ggtune::Parameter>,

    /// Random number generator seed for reproducible runs.
    #[structopt(long, default_value = "7861")]
    seed: usize,

    #[structopt(flatten)]
    minimizer: ggtune::Minimizer,

    /// How the objective value should be transformed (lin/linear or log/ln/logarithmic).
    #[structopt(long, default_value = "linear")]
    transform_objective: ggtune::Projection,

    /// Whether 32-bit numbers should be used. Faster, but has numeric stability problems.
    #[structopt(long)]
    use_32: bool,

    /// A CSV into which evaluation results are written.
    /// Overwrites the file contents!
    #[structopt(long)]
    write_csv: Option<std::path::PathBuf>,

    #[structopt(subcommand)]
    objective: CliObjective,
}

#[derive(Debug, StructOpt)]
struct CliCommandFunction {
    /// Random number generator seed for reproducible runs.
    #[structopt(long, default_value = "7861")]
    seed: usize,

    #[structopt(flatten)]
    function: CliBenchFunction,

    /// Sample at which the function shall be evaluated.
    args: Vec<f64>,
}

#[derive(Debug, StructOpt)]
enum CliObjective {
    /// As the objective function, execute an external program.
    #[structopt(name = "command")]
    Command {
        /// The shell command to invoke for each sample.
        /// Can substitute parameter values by name.
        /// E.g. `./objective "{x1}" --param={x2}`
        #[structopt(name = "objective-command", min_values = 1)]
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
    fn into_objective<'a, A>(
        self,
        space: &ggtune::Space,
    ) -> Box<dyn ggtune::ObjectiveFunction<A> + 'a>
    where
        A: ObjectiveValue,
        f64: num_traits::AsPrimitive<A>,
    {
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
    SumAbs,
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
            "sum-abs" => BenchFn::SumAbs,
            _ => bail!("expected a benchmark function name, got: {:?}", name),
        })
    }
}

trait ObjectiveValue
where
    Self: num_traits::Float
        + num_traits::FloatConst
        + num_traits::Signed
        + Into<f64>
        + From<u16>
        + From<i16>
        + Default
        + std::fmt::Display
        + std::str::FromStr
        + 'static,
{
}

impl ObjectiveValue for f32 {}
impl ObjectiveValue for f64 {}

impl<A> ggtune::ObjectiveFunction<A> for CliBenchFunction
where
    A: ObjectiveValue,
    f64: num_traits::AsPrimitive<A>,
{
    fn run(&self, xs: &[ggtune::ParameterValue], rng: &mut ggtune::RNG) -> (A, A) {
        use ggtune::benchfn;
        use num_traits::AsPrimitive as _;

        let xs = xs.iter().map(|&x| (x.to_f64().as_())).collect_vec();
        match self.function {
            BenchFn::GoldsteinPrice | BenchFn::Easom { .. } | BenchFn::Himmelblau => assert_eq!(
                xs.len(),
                2,
                "objective function requires exactly two dimensions"
            ),
            BenchFn::Sphere
            | BenchFn::Rastrigin { .. }
            | BenchFn::Rosenbrock
            | BenchFn::Onemax
            | BenchFn::SumAbs => {}
        };
        use ndarray::Array;
        let y: A = match self.function {
            BenchFn::Sphere => benchfn::sphere(Array::from(xs)),
            BenchFn::GoldsteinPrice => benchfn::goldstein_price(xs[0], xs[1]),
            BenchFn::Easom { amplitude } => benchfn::easom(xs[0], xs[1], amplitude.as_()),
            BenchFn::Himmelblau => benchfn::himmelblau(xs[0], xs[1]),
            BenchFn::Rastrigin { amplitude } => {
                benchfn::rastrigin(Array::from(xs), amplitude.as_())
            }
            BenchFn::Rosenbrock => benchfn::rosenbrock(Array::from(xs)),
            BenchFn::Onemax => benchfn::onemax(Array::from(xs)),
            BenchFn::SumAbs => benchfn::sum_abs(Array::from(xs)),
        };
        (rng.normal(y.into(), self.noise).as_(), Default::default())
    }
}

fn main() {
    let args = CliApp::from_args();
    if args.verbose {
        println!("args: {:#?}", args);
    }
    let result: Result<(), _> = match args.command {
        CliCommand::Run(run) => {
            if run.use_32 {
                command_run::<f32>(run, args.quiet)
            } else {
                command_run::<f64>(run, args.quiet)
            }
        }
        CliCommand::Function(function) => command_function(function),
    };

    if let Err(err) = result {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}

fn command_run<A>(cfg: CliCommandRun, quiet: bool) -> Result<(), failure::Error>
where
    A: ggtune::Scalar + ObjectiveValue,
    f64: num_traits::AsPrimitive<A>,
{
    use ggtune::{Estimator, EstimatorGPR, MinimizerArgs, ObjectiveFunction};

    let CliCommandRun {
        param: params,
        seed,
        minimizer,
        objective,
        transform_objective,
        use_32: _use_32,
        write_csv,
    } = cfg;

    ensure!(
        !params.is_empty(),
        "Option --param must be provided at least once"
    );

    let mut space = ggtune::Space::new();
    for param in params {
        space.add_parameter(param.clone());
    }

    let mut rng = ggtune::RNG::new_with_seed(seed);

    let objective: Box<dyn ObjectiveFunction<A>> = objective.into_objective(&space);

    let mut opened_csv_file = None;
    let mut args = MinimizerArgs::<A, EstimatorGPR>::default();

    if !quiet {
        args.output
            .add_human_readable_individuals(std::io::stdout(), &space);
    }

    if let Some(file) = write_csv {
        opened_csv_file.replace(
            std::fs::File::create(file)
                .map_err(|err| format_err!("cannot open CSV file: {}", err))?,
        );
        args.output
            .add_csv_writer(opened_csv_file.as_mut().unwrap(), &space)?;
    }

    args.estimator =
        Some(<EstimatorGPR as Estimator<A>>::new(&space).y_projection(transform_objective));

    let result = minimizer
        .minimize(objective.as_ref(), space.clone(), &mut rng, args)
        .map_err(|err| format_err!("error during minimization: {}", err))?;

    let suggestion_location = space
        .params()
        .iter()
        .zip_eq(result.suggestion_location())
        .map(|(param, value)| {
            json!({
                "name": param.name(),
                "type": match value {
                    ggtune::ParameterValue::Real(_) => "real",
                    ggtune::ParameterValue::Int(_) => "int",
                },
                "value": match value {
                    ggtune::ParameterValue::Real(x) => json!(x),
                    ggtune::ParameterValue::Int(x) => json!(x),
                },
            })
        })
        .collect_vec();
    let suggestion_statistics = result.suggestion_statistics();

    println!(
        "optimization result: {:#}",
        json!({
            "location": suggestion_location,
            "mean": suggestion_statistics.mean(),
            "std": suggestion_statistics.std(),
            "median": suggestion_statistics.median(),
            "cv": suggestion_statistics.cv(),
            "iqr": suggestion_statistics.iqr(),
            "q1": suggestion_statistics.q13().0,
            "q3": suggestion_statistics.q13().1,
        })
    );

    Ok(())
}

fn command_function(function: CliCommandFunction) -> Result<(), failure::Error> {
    let CliCommandFunction {
        seed,
        function,
        args,
    } = function;

    let sample = args.into_iter().map(Into::into).collect_vec();
    let mut rng = ggtune::RNG::new_with_seed(seed);

    let (value, _cost): (f64, f64) =
        ggtune::ObjectiveFunction::run(&function, sample.as_slice(), &mut rng);

    println!("{}", value);

    Ok(())
}
