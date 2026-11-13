PYTHON ?= python3
RESULTS_ROOT ?= runs

.PHONY: help install smoke main convergence memory ablations tinytl time_spec all clean-runs

help:
	@echo "MemFLoRA - reproduction targets"
	@echo
	@echo "  make install      install the package and its pinned dependencies"
	@echo "  make smoke        tiny end-to-end run to check the pipeline (minutes)"
	@echo
	@echo "  make main         main accuracy experiments        (6 runs)"
	@echo "  make convergence  adaptation-step curves           (2 runs)"
	@echo "  make memory       SRAM / performance profiling     (4 runs)"
	@echo "  make ablations    design ablations                 (4 runs)"
	@echo "  make tinytl       TinyTL-style comparison          (3 runs)"
	@echo "  make time_spec    adaptation-time metrics          (2 runs)"
	@echo "  make all          every experiment above          (21 runs)"
	@echo
	@echo "  make clean-runs   delete $(RESULTS_ROOT)/"
	@echo
	@echo "Full runs need a GPU and take a long time; output goes to $(RESULTS_ROOT)/."
	@echo "Datasets are not shipped - see README.md."

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	PYTHON="$(PYTHON)" RESULTS_ROOT="$(RESULTS_ROOT)/smoke" bash scripts/00_smoke_test.sh

main convergence memory ablations tinytl time_spec all:
	PYTHON="$(PYTHON)" RESULTS_ROOT="$(RESULTS_ROOT)" bash scripts/run_all.sh $@

clean-runs:
	rm -rf -- "$(RESULTS_ROOT)"
	@echo "removed $(RESULTS_ROOT)/"
