SPOTGP_SIF ?= $(HOME)/containers/spotgp.sif
CONFIG     ?= configs/example.yaml
PORT       ?= 8501

.PHONY: run run-container submit validate test shell app clean

run:
	python scripts/run_fit.py $(CONFIG)

run-container:
	apptainer exec --bind $(PWD):/work $(SPOTGP_SIF) python /work/scripts/run_fit.py /work/$(CONFIG)

submit:
	mkdir -p logs
	sbatch scripts/run_fit.slurm $(CONFIG)

validate:
	python scripts/run_fit.py $(CONFIG) --validate

test:
	pytest -v

app:
	streamlit run scripts/app.py --server.port=$(PORT)

shell:
	apptainer shell --bind $(PWD):/work $(SPOTGP_SIF)

clean:
	rm -rf logs/*.out logs/*.err results/*/ results/*.h5 metrics.json
