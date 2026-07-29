SPOTGP_SIF ?= $(HOME)/containers/spotgp.sif
SCRIPTS    ?= $(dir $(abspath $(lastword $(MAKEFILE_LIST))))scripts
CONFIG     ?= configs/example.yaml
PORT       ?= 8501
PROJECT_DIR ?=

# Flags forwarded when PROJECT_DIR is set
_PD_FLAG = $(if $(PROJECT_DIR),--project-dir $(PROJECT_DIR))

.PHONY: init run run-container submit validate test shell app docs serve clean

init:
	python $(SCRIPTS)/init_project.py $(PROJECT_DIR)

run:
	python $(SCRIPTS)/run_fit.py $(CONFIG) $(_PD_FLAG)

run-container:
	apptainer exec --bind $(PWD):/work $(SPOTGP_SIF) python /work/scripts/run_fit.py /work/$(CONFIG)

submit:
	mkdir -p logs
	sbatch scripts/run_fit.slurm $(CONFIG)

validate:
	python $(SCRIPTS)/run_fit.py $(CONFIG) --validate $(_PD_FLAG)

test:
	pytest -v

app:
	streamlit run $(SCRIPTS)/app.py --server.port=$(PORT) $(if $(PROJECT_DIR),-- --project-dir $(PROJECT_DIR))

shell:
	apptainer shell --bind $(PWD):/work $(SPOTGP_SIF)

docs:
	mkdocs build

serve:
	mkdocs serve

clean:
	rm -rf logs/*.out logs/*.err results/*/ results/*.h5 metrics.json
