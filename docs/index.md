# spotgp-project

A template for running [spotgp](https://github.com/jbirky/spotgp) Gaussian Process fits on stellar light curves, with config-driven experiments, HDF5 results, DVC pipelines, and optional experiment tracking.

## Overview

spotgp models stellar variability caused by starspots using analytic GP kernels derived from spot evolution models. This project template provides the scaffolding to run reproducible fits on TESS, Kepler, and K2 light curves.

**Key features:**

- YAML-driven experiment configs
- MAP optimization and MCMC sampling (NUTS / nested sampling)
- Interactive Streamlit explorer for tuning model parameters
- Self-contained HDF5 result files with embedded configs
- DVC pipelines for reproducibility and experiment comparison

## Getting started

- [**Installation**](installation.md) — setup, dependencies, HPC containers, project structure
- [**Interactive Explorer**](gui-guide.md) — using the Streamlit GUI to tune models and run fits
- [**YAML Config Guide**](yaml-guide.md) — config reference for the command-line runner and DVC pipelines

## Projects

Browse the [Projects](projects/index.md) gallery to see analyses contributed by the team.

To add your own, copy the [project template](https://github.com/jbirky/spotgp-project/blob/main/docs/projects/_template.md) and submit a pull request.
