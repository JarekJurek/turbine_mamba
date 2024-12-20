# Turbine Momentum Prediction with Mamba Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains code for predicting wind turbine blade pitch momentum using advanced machine learning
architectures. The project primarily leverages **Mamba**, a state-of-the-art selective state-space model (SSM) for
sequence modeling.

## Project Overview

Wind turbines, especially offshore installations, operate in challenging environments where efficient blade pitch
control is crucial. This project explores the application of Mamba to predict windmill blade pitch momentum,
demonstrating its ability to accurately estimate momentum components with minimal deviation. The results highlight the
potential of Mamba to improve wind turbine performance and reduce maintenance costs.

## Data

The dataset used in this project was generated using OpenFAST, a high-fidelity simulator for wind turbine dynamics. It
represents the NREL 5 MW Wind Turbine under various wind conditions. The dataset includes input features such as blade
pitch angles, rotor speed, and wind speed, as well as output features corresponding to the wind load on each blade.
These inputs and outputs are critical for modeling blade pitch momentum, enabling accurate predictions and analysis of
turbine performance in diverse scenarios.

*The `FFNN` reference implementation is available on the branch `FFNN`.*

## Authors

- Filip Blaafjell
- Grzegorz Pawlak
- Malfridur Eiriksdottir
- Matteo D’Souza

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         turbine_mamba and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── turbine_mamba   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes turbine_mamba a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- PyTorch 1.10 or higher
- Transformers library
- Other dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/JarekJurek/turbine_mamba.git
cd turbine_mamba
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

To train and evaluate the Mamba model:

```bash
python3 main.py
```

---

This repository follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project
structure.
