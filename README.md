# ggtune

The ggtune tool is a global optimization engine
for expensive, noisy, black-box functions.
It uses a hybrid algorithm combining Bayesian Optimization
and Evolutionary Algorithms.

## `ggtune`

Usage: `ggtune [OPTIONS] COMMAND ...`

Options:

* **`--help`**: show a help message.
* **`--version`**: show the program version.
* **`--verbose`**: enable verbose output.
* **`--quiet`**: output less information.

Commands:

* **`help`**: show a help help message.
* **`run`**: run the ggtune minimizer.
* **`function`**: evaluate a built-in benchmark function.

## `ggtune run`

Usage: `ggtune run [OPTIONS] OBJECTIVE ...`

Run the minimizer.
This requires defining the objective function that shall be optimized,
the parameter space that can be explored,
and choosing output settings.

General options:

* **`--param PARAMETER`**:
  adds a parameter/dimension to the input space.
  Can be provided multiple times.
  Each parameter has a name, type, and possibly other arguments.
* **`--seed SEED`**:
  RNG seed for reproducible runs (is constant by default).
* **`--transform-objective PROJECTION`**:
  specify transformations of the objective value (default linear).
  * `lin`, `linear`:
    no extra warping of the objective value, though it is rescaled internally.
  * `log`, `ln`, `logarithmic`:
    use a logarithmic transformation for the objective,
    which can help with many target functions.
    However, interpretation of the internal model becomes more difficult.
    The data is rescaled internally so that it can handle negative values.
* **`--known-optimum OBJECTIVE`**:
  Known optimum (lower bound) for the objective value.
  Serves as a bias for the surrogate model.
  Example: the known optimum of an objective representing a distance is at least zero.
  By default, a lower bound is inferred.
* **`--use-32`**:
  use 32-bit floats internally.
  Leads to numeric stability problems.
  Really not recommended.
* **`--write-csv PATH`**:
  write evaluation results to the given file in a CSV format.
  The file will be created or overwritten.

Minimizer options:

* **`--popsize N`**:
  how many samples are evaluated per generation (default 10).
* **`--initial N`**:
  how may initial samples should be evaluated
  before model-guided acquisition takes over (default 10).
* **`--validation N`**:
  how many samples are taken for validation of the optimum (default 1).
* **`--max-nevals N`**:
  how many samples may be evaluated in total (default 100).
* **`--relscale-initial STD`**:
  standard deviation for creating new samples,
  as a fraction of each parameters range (default 0.3).
* **`--relscale-attenuation RATIO`**:
  factor by which relscale is shrunk per generation.
  Smaller values converge faster,
  but possibly not to the optimum.
  (default 0.9).
* **`--select-via FITNESS`**:
  fitness function used to select which samples proceed to the next generation
  (default observation).
* **`--fmin-via FITNESS`**:
  fitness function used to estimate the minimum function value
  (default prediction).
* **`--competition-rate RATE`**:
  mean rate at which rejected individuals are given a second chance
  when selecting between parents and offspring for a new generation.
  Towards zero, there is only offspring–parent competition.
  Towards one, all individuals compete against each other.
  (default 0.25).

Parameters:

All parameter specifications start with a NAME for the parameter,
next the parameter type is given.
Depending on the type there may be further arguments.
Items in a parameter specification are separated by spaces and colons `:`.

* **`NAME real LO HI`**:
  a real-valued parameter between LO and HI (inclusive).
  Values are assumed to follow an uniform distribution.
* **`NAME int LO HI`**:
  an integer-valued parameter between LO and HI (inclusive),
  otherwise similar to `real`.
* **`NAME logreal LO HI [OFFSET]`**:
  a real-value parameter between LO and HI (inclusive).
  Values are assumed to follow a logarithmic distribution.
  The OFFSET defaults to zero, and must be chosen so that `LO + OFFSET > 0`.
* **`NAME logint LO HI [OFFSET]`**:
  an integer-valued parameter between LO and HI (inclusive),
  otherwise similar to `logreal`.

Fitness functions:

* `posterior`, `prediction`, `model`:
  use the trained model to determine fitness.
  This may remove noise, but can also smooth over the actual value.
* `observation`:
  use the sampled objective value to determine fitness.
  This value may include noise.

Objective function:

* **`command -- CMD ARG...`**:
  run an external program as the objective function.
  The command must return a real-valued objective value on the last line of output,
  other output is ignored.
  Parameters can be interpolated into the arguments by name,
  e.g. `./objective "{x1}" --param={x2}`.
  Additionally, the `SEED` variable inserts a random 32-bit unsigned integer value.

* **`function [--noise=STD] NAME`**:
  use a built-in benchmark function as the objective function,
  see `ggtune function`.

## `ggtune function`

Usage: `ggtune function [OPTIONS] NAME ARGS...`

Evaluate an example function at the location given by ARGS,
and possibly add noise.
Some example functions have a fixed dimension,
others can take any number of arguments.

Options:

* **`--seed SEED`** RNG seed for reproducible runs, is used for noise
* **`--noise STD`** standard deviation for additive Gaussian noise.
  Defaults to zero (no noise).

Available functions:

* **`sphere`**
* **`goldstein-price`**
* **`easom`**
* **`himmelblau`**
* **`rastrigin`**
* **`rosenbrock`**
* **`onemax`**
* **`sum-abs`**

## Installation

While this software is an ordinary Rust package,
it is advised to not run `cargo` directly,
at least the first time:
the software builds its own OpenBLAS library with custom options,
which requires environment variables to set in the Makefile
(until [rust-lang/cargo#4121](https://github.com/rust-lang/cargo/issues/4121)
is resolved).
Therefore:

* `make install` to install the software locally, or
* `make release` to compile a statically-linked executable into `target/release/ggtune`
