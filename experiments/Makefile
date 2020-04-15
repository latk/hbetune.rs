IRACE ?= irace
HBETUNE ?= hbetune
export IRACE
export HBETUNE

RESULTS ?= results

SHELL = bash
.SHELLOPTS: -euo pipefail

ALL_EXPERIMENTS = \
	rosenbrock rosenbrock+noise1 rosenbrock+noise10 rosenbrock+noise10+nevals200 \
	goldsteinprice goldsteinprice+noise1 goldsteinprice+noise10 goldsteinprice+noise10+nevals200 \
	himmelblau himmelblau+noise1 himmelblau+noise10 himmelblau+noise10+nevals200 \
	rastrigin $(foreach suffix,+noise1 +noise10 +noise10+nevals200,rastrigin$(suffix)) \
	hberosenbrock hberastrigin acotsp

.PHONY: all summary $(ALL_EXPERIMENTS)

all: $(ALL_EXPERIMENTS)

SUMMARY_PARTS = $(RESULTS)/summary.tex $(ALL_EXPERIMENTS:%=$(RESULTS)/%.pdf)
summary: | ./summarize-experiments $(SUMMARY_PARTS)
$(SUMMARY_PARTS): summarize-experiments $(wildcard $(RESULTS)/*.validation)
	./summarize-experiments $(ALL_EXPERIMENTS:%=$(RESULTS)/%)

# $(call make_seeds, N, METASEED)
make_seeds = $(shell python3 -c 'import sys, random as R; R.seed(sys.argv[2], version=2); [print(R.randint(1, 1<<31)) for _ in range(int(sys.argv[1]))]' $(1) $(2))

# $(call experiment_result_files, NAME, N, EXTRA_ALGOS)
experiment_result_files = \
	$(shell ./generate-experiments $(RESULTS) $(1) $(2) $(3))

# syntax: RESULTS/NAME[+NOISE][+NEVALS]-...-seedSEED-ALGORITHM.SUFFIX
make_experiment_command = \
  $(foreach base,$(notdir $(basename $(1))), \
		$(call make_experiment_command_inner,\
			$(basename $(1)),\
			$(subst +, ,$(firstword $(subst -, ,$(base)))),\
			$(subst -, ,$(base)))) \
  $(call make_experiment_command_subcommand,$(1))

# $(call make_experiment_command_subcommand, SUFFIX)
# returns: "validation" for validation files, the algo name otherwise
make_experiment_command_subcommand = \
	$(if $(filter .validation, $(suffix $(1))), \
		validation, \
		$(lastword $(subst -, ,$(basename $(1)))))

# $(call make_experiment_command_inner, PREFIX, NAMEPARTS, OTHERPARTS)
make_experiment_command_inner = \
	env \
	PREFIX=$(strip $(1)) \
	SEED=$(subst seed,,$(filter seed%,$(3))) \
	NOISE=$(subst noise,,$(filter noise%,$(2))) \
	NEVALS=$(subst nevals,,$(filter nevals%,$(2))) \
	./experiment-$(subst _,-,$(firstword $(2)))

# experiment declarations

rosenbrock: $(call experiment_result_files,rosenbrock,100,hbetunelog)
rosenbrock+noise1: $(call experiment_result_files,rosenbrock+noise1,100,hbetunelog)
rosenbrock+noise10: $(call experiment_result_files,rosenbrock+noise10,100,hbetunelog)
rosenbrock+noise10+nevals200: $(call experiment_result_files,rosenbrock+noise10+nevals200,100,hbetunelog)

goldsteinprice: $(call experiment_result_files,goldsteinprice,100,hbetunelog)
goldsteinprice+noise1: $(call experiment_result_files,goldsteinprice+noise1,100,hbetunelog)
goldsteinprice+noise10: $(call experiment_result_files,goldsteinprice+noise10,100,hbetunelog)
goldsteinprice+noise10+nevals200: $(call experiment_result_files,goldsteinprice+noise10+nevals200,100,hbetunelog)

himmelblau: $(call experiment_result_files,himmelblau,100,hbetunelog)
himmelblau+noise1: $(call experiment_result_files,himmelblau+noise1,100,hbetunelog)
himmelblau+noise10: $(call experiment_result_files,himmelblau+noise10,100,hbetunelog)
himmelblau+noise10+nevals200: $(call experiment_result_files,himmelblau+noise10+nevals200,100,hbetunelog)

rastrigin: $(call experiment_result_files,rastrigin,100,hbetunelog)
rastrigin+noise1: $(call experiment_result_files,rastrigin+noise1,100,hbetunelog)
rastrigin+noise10: $(call experiment_result_files,rastrigin+noise10,100,hbetunelog)
rastrigin+noise10+nevals200: $(call experiment_result_files,rastrigin+noise10+nevals200,100,hbetunelog)

hberosenbrock: $(call experiment_result_files,hberosenbrock,30)
hberastrigin: $(call experiment_result_files,hberastrigin,30)
acotsp: ACOTSP-1.03/acotsp $(call experiment_result_files,acotsp,30)

ACOTSP-1.03/%:
	$(MAKE) -C ACOTSP-1.03 $*

# generic rules:

$(RESULTS)/%-hbetune.json $(RESULTS)/%-hbetune.csv:
	@mkdir -p $(RESULTS)
	$(call make_experiment_command,$@)

$(RESULTS)/%-hbetunelog.json $(RESULTS)/%-hbetunelog.csv:
	@mkdir -p $(RESULTS)
	$(call make_experiment_command,$@)

$(RESULTS)/%-irace.rdata:
	@mkdir -p $(RESULTS)
	$(call make_experiment_command,$@)

$(RESULTS)/%.validation : $(RESULTS)/%.json
	$(call make_experiment_command,$@)

$(RESULTS)/%.debug:
	echo $(call make_experiment_command,$@)

$(RESULTS)/%.json : $(RESULTS)/%.rdata
	./convert-from-rdata $< $(<:.rdata=.csv) $(<:.rdata=.json)
