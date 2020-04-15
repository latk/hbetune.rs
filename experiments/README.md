# Benchmarks and Experiments

This directory contains a couple of benchmarks that pit hbetune against irace.
The experiments are controlled via the `Makefile`
and produce output in the `results` directory.

Result files follow the naming schema `results/EXPERIMENT-seed1234-TUNER.json` etc.

Careful: running a practical experiment takes about an hour,
or roughly 5h for all the experiments.
(Reference: a modern laptop CPU)

## Summaries

With `make summaries`, all experiments will be summarized,
which produces the figures and tables for the paper.
In particular:

* `results/EXPERIMENT.pdf`:
  the boxplot for the experiment
* `results/EXPERIMENT.parameters.tex`:
  parameter space + results for practical experiments
* `results/summary.tex`:
  the big results table

## Dependencies

For experiments:

* build `hbetune` and put it into the PATH
* install the `irace` R library
* install command line tools such as `jq`

For summaries:

* install Python 3.7
* install the following Python libraries:
  `scipy.stats`, `click`, `matplotlib`, `numpy`, `pandas`
  (but there might be incompatibilities with newer versions)
