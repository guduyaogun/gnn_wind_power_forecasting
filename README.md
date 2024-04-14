# Physics-Induced Graph Neural Network for Probabilistic Spatio-Temporal Forecasting of Wind Power

In this project, we develop physics-induced graph neural networks for spatio-temporal, probabilistic wind power forecasting. The newly developed methods will be benchmarked against other state-of-the-art forecasting models.

## Getting Started

### Installing Dependencies

To install the project's dependencies, create a new conda environment, activate it, and first run

```
conda install pip
```

Then, in the project folder root, run

```
pip install -r requirements.txt
```

Due to some conflicting package dependencis, we recommend to then run separately

```
pip install torch-geometric-temporal --no-deps
```

### Running a Model

## Data

The datasets used here are

- 10min Supervisory Control and Data Acquisition (SCADA) data from the six wind turbines at the Kelmarsh wind farm [[1]](#1), available under a Creative Commons Attribution 4.0 International license, and
- 5min SCADA data from 2022 to 2023 for 67 Australian wind farms, retrieved from the Nemweb archive from AEMO [[2]](#2), which is free to use for any purpose, when accurately and appropriately attributing AEMO as the author, see the [AEMO Copyright Permissions](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-data-nemweb).

## Acknowledgment

We appreciate the following GitHub repository for its publicly available code and methods, which we use as benchmark methods for our architecture:

https://github.com/LarsBentsen/FFTransformer

## References

<a id="1">[1]</a>
Plumley, C. (2022).
Kelmarsh wind farm data (0.1.0).
Zenodo.
https://doi.org/10.5281/zenodo.8252025

<a id="2">[2]</a>
AEMO (2024).
Nemweb Archive Reports.
National Electricity Market Web.
https://nemweb.com.au/Reports/Archive/
