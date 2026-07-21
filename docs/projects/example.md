# TIC 441420236 — Trapezoidal Spot Model

## Abstract

TIC 441420236 is a short-period variable observed by TESS in Sectors 1 and 2. We fit a Gaussian Process model using a symmetric trapezoidal spot evolution envelope and full-geometry visibility function to characterize the stellar rotation period and spot properties. The MAP optimization recovers a rotation period consistent with the periodogram peak, and nested sampling provides posterior constraints on the spot lifetime and differential rotation.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Envelope | TrapezoidSymmetricEnvelope |
| Visibility | VisibilityFunction |
| Latitude distribution | Uniform |
| Sampler | dynesty (nlive=500) |

<details>
<summary>Full config YAML</summary>

```yaml
star_name: TIC 441420236
seed: 42

data:
  sectors: [1, 2]
  normalize: true
  zero_mean: false
  downsample: 1

model:
  envelope: TrapezoidSymmetricEnvelope
  envelope_params:
    lspot: 15.0
    tau_spot: 5.0
  visibility: VisibilityFunction
  visibility_params:
    peq: 3.3
    kappa: 0.1
    inc: 1.31
  sigma_k: 0.005

bounds:
  peq: [1.0, 10.0]
  kappa: [0.0, 0.5]
  log_sigma_k: [-5.0, -1.0]

solver:
  kernel_type: analytic
  matrix_solver: cholesky_banded

fitting:
  map:
    nopt: 10
  sampling:
    sampler: dynesty
    nlive: 500

output:
  save_dir: results
```

</details>

## Results

| Parameter | MAP | Posterior median | 1-sigma |
|-----------|-----|-----------------|---------|
| P_eq (days) | 3.31 | 3.29 | 0.05 |
| kappa | 0.12 | 0.11 | 0.03 |
| log sigma_k | -2.35 | -2.40 | 0.15 |

## Figures

<!-- Add your plots here, e.g.:
![ACF comparison](img/tic441420236_acf.png)
![GP prediction](img/tic441420236_prediction.png)
-->

## Notes

This is an example project entry. Replace the results above with your actual fitted values and add diagnostic plots to `docs/projects/img/`.
