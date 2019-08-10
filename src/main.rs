extern crate ggtune;
extern crate strfmt;
#[macro_use]
extern crate structopt;
extern crate itertools;
extern crate ndarray;
#[macro_use]
extern crate failure;

use itertools::Itertools as _;
use ndarray::prelude::*;
use structopt::StructOpt as _;

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

    /// Name of the function. (sphere)
    function: BenchFn,
}

impl CliObjective {
    fn into_objective(self, space: &ggtune::Space) -> Box<dyn ggtune::ObjectiveFunction<f64>> {
        match self {
            CliObjective::Command { objective_command } => Box::new(RunCommandAsObjective {
                cli_template: objective_command,
                space: space.clone(),
            }),
            CliObjective::Function(f) => Box::new(f),
        }
    }
}

#[derive(Debug)]
enum BenchFn {
    Sphere,
}

impl std::str::FromStr for BenchFn {
    type Err = failure::Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Ok(match name {
            "sphere" => BenchFn::Sphere,
            _ => bail!("expected sphere, got: {:?}", name),
        })
    }
}

impl ggtune::ObjectiveFunction<f64> for CliBenchFunction {
    fn run(&self, xs: ndarray::ArrayView1<f64>, rng: &mut ggtune::RNG) -> (f64, f64) {
        use ggtune::benchfn;
        let y = match self.function {
            BenchFn::Sphere => benchfn::sphere(xs),
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

    let objective: Box<dyn ggtune::ObjectiveFunction<f64>> = objective.into_objective(&space);

    let args = ggtune::MinimizerArgs::<f64, ggtune::EstimatorGPR>::default();
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

struct RunCommandAsObjective {
    cli_template: Vec<String>,
    space: ggtune::Space,
}

impl ggtune::ObjectiveFunction<f64> for RunCommandAsObjective {
    fn run<'a>(&self, xs: ArrayView1<'a, f64>, _rng: &'a mut ggtune::RNG) -> (f64, f64) {
        let template = self.cli_template.iter().map(String::as_ref).collect_vec();
        let xs = xs.to_vec();

        run_command_as_objective(&self.space, template.as_slice(), xs.as_slice())
    }
}

fn run_command_as_objective(space: &ggtune::Space, template: &[&str], xs: &[f64]) -> (f64, f64) {
    use std::collections::HashMap;
    use std::process::*;

    let mut args = HashMap::new();
    for (param, value) in space.params().iter().zip_eq(xs) {
        args.insert(param.name().to_owned(), value);
    }

    let processed_cmd_args = template
        .iter()
        .map(|template| {
            strfmt::strfmt(template, &args)
                .expect("filling in objective command placeholders must succeed")
        })
        .collect_vec();
    let mut processed_cmd_args = processed_cmd_args
        .iter()
        .map(AsRef::<std::ffi::OsStr>::as_ref);

    let mut command = Command::new(
        processed_cmd_args
            .next()
            .expect("objective command needs a command name"),
    );

    for arg in processed_cmd_args {
        command.arg(arg);
    }

    let output = command
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .expect("objective command failed to execute");

    assert!(
        output.status.success(),
        "objective command status was nonzero: {}",
        output.status,
    );

    let output = String::from_utf8(output.stdout).expect("objective command output must be UTF-8");

    let last_line = output
        .lines()
        .last()
        .expect("objective command: at least one line of output is required");

    let mut items = last_line.split_whitespace();

    let value = items
        .next()
        .expect("objective command: output must contain value")
        .parse::<f64>()
        .expect("objective command: value must parse as a number");

    let cost = items
        .next()
        .unwrap_or("0")
        .parse::<f64>()
        .expect("objective command: cost must parse as a number");

    assert!(
        items.next().is_none(),
        "objective command: last output line can only contain two values",
    );

    (value, cost)
}
