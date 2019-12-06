CARGO ?= cargo

# needed to force the build of openblas into single-threaded mode:
export USE_THREAD = 0

.PHONY: all lint fmt test release run

all:
	# please select a target

lint:
	$(CARGO) clippy

fmt:
	$(CARGO) fmt

test:
	$(CARGO) test $(TEST)

release:
	$(CARGO) build --release

run:
	$(CARGO) run $(ARGS)

flamegraph.svg: $(wildcard src/** tests/**)
	$(CARGO) flamegraph --dev --test minimize_test -- --test-threads 1
