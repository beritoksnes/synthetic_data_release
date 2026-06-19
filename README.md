# Privacy evaluation framework for synthetic data publishing
A practical framework to evaluate the privacy-utility tradeoff of synthetic data publishing 

Based on "Synthetic Data - Anonymisation Groundhog Day, Theresa Stadler, Bristena Oprisanu, and Carmela Troncoso, [arXiv](https://arxiv.org/abs/2011.07018), 2020"

# Attack models
The module `attack_models` so far includes

A privacy adversary to test for privacy gain with respect to linkage attacks modelled as a membership inference attack `MIAAttackClassifier`.

A simple attribute inference attack `AttributeInferenceAttack` that aims to infer a target's sensitive value given partial knowledge about the target record

# Generative models
The module `generative_models` so far includes:   
- `IndependentHistogram`: An independent histogram model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `BayesianNet`: A generative model based on a Bayesian Network adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `PrivBayes`: A differentially private version of the BayesianNet model adapted from [Data Responsibly's DataSynthesiser](https://github.com/DataResponsibly/DataSynthesizer)
- `CTGAN`: A conditional tabular generative adversarial network that integrates the CTGAN model from [CTGAN](https://github.com/sdv-dev/CTGAN)  
- `PATE-GAN`: A differentially private generative adversarial network adapted from its original implementation by the [MLforHealth Lab](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/)

# Setup

## Docker Distribution

For your convenience, Synthetic Data is also distributed as a ready-to-use Docker image containing Python 3.9 and CUDA 11.4.2, along with all dependencies required by Synthetic Data, including jupyter notebook to visualize and analyse the results.

**Note:** This distribution includes CUDA binaries, before downloading the image, ensure to read [its EULA](https://docs.nvidia.com/cuda/eula/index.html) and to agree to its terms.

Pull the image and run a container (and bind a volume where you want to save the data):

```
docker pull springepfl/synthetic-data:latest
docker run -it --rm -v "$(pwd)/output:/output" -p 8888:8888 springepfl/synthetic-data
```

The Synthetic Data directory is placed at the root directory of the container.
```
cd /synthetic_data_release
```

You should now be able to run the examples without encountering any problems, and you should be able to visualize the results with Jupyter by running
```
jupyter notebook --allow-root --ip=0.0.0.0
```

and opening the notebook with your favourite web browser at the url `http://127.0.0.1:8888/?token=<authentication token>`.


## Direct Installation

### Requirements
The framework and its building blocks have been developed and tested under Python 3.9.

We recommend creating a virtual environment for installing all dependencies and running the code:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note (fork-specific):** The original README installs numpy separately before other dependencies due to API compatibility issues. In this fork, `requirements.txt` has been updated with fully pinned versions that install correctly in one step. See [Known Issues](#known-issues) if you encounter problems.

### Dependencies
The `CTGAN` model depends on the original CTGAN package from [sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN).

> **Note (fork-specific):** The original README references a spring-epfl fork of CTGAN. This fork uses the upstream sdv-dev CTGAN instead, with manual fixes to `setup.py` to resolve version conflicts.

Clone and install CTGAN in editable mode:
```bash
git clone https://github.com/sdv-dev/CTGAN
cd CTGAN
```

Before installing, edit `setup.py` to fix two version conflicts:
- Relax the `scikit-learn` upper bound from `<0.23` to `<0.25`
- Pin `torchvision` to `==0.10.1` (required for compatibility with `torch==1.9.1`)

Then install:
```bash
pip install torchvision==0.10.1
pip install -e . --no-deps
```

To test your installation, run from within your virtualenv:
```python
import ctgan
```

### Known Issues

The following issues were encountered during setup on Python 3.9 without Docker and are resolved in this fork's `requirements.txt`.

**protobuf / TensorFlow import error**
TensorFlow 2.6.2 is incompatible with newer protobuf versions. Fixed by pinning:
```
protobuf==3.20.3
```

**`%matplotlib inline` fails in notebook**
Caused by a mismatch between `matplotlib==3.4.3` and a newer `matplotlib-inline`. Fixed by pinning:
```
matplotlib-inline==0.1.3
```

**`No module named 'pkg_resources'`**
Caused by a broken `setuptools` installation. Fixed by running:
```bash
pip install --force-reinstall setuptools
```

**Harmless warnings (safe to ignore)**
- `libcudart.so.11.0: cannot open shared object file` — no GPU available, CPU fallback is used automatically
- `disable_resource_variables ... deprecated` — TF 2.6 compatibility shim, does not affect results
- `ctgan has requirement pandas<0.26` — CTGAN's pandas pin is an old artifact; 1.3.5 works fine
- `tensorflow requires typing-extensions~=3.7.4` — does not cause issues in practice with 4.x

# Example runs

## Membership Inference Attack (MIA)
To run a privacy evaluation with respect to the privacy concern of linkability you can run

```
python3 linkage_cli.py -D data/texas -RC tests/linkage/runconfig.json -O tests/linkage
```

The results file produced after successfully running the script will be written to `tests/linkage` and can be parsed with the function `load_results_linkage` provided in `utils/analyse_results.py`. 

## Attribute Inference Attack (AIA)
To run a privacy evaluation with respect to the privacy concern of inference you can run

```
python3 inference_cli.py -D data/texas -RC tests/inference/runconfig.json -O tests/inference
```

The results file produced after successfully running the script can be parsed with the function `load_results_inference` provided in `utils/analyse_results.py`.


## Utility evaluation
To run a utility evaluation with respect to a simple classification task as utility function run

```
python3 utility_cli.py -D data/texas -RC tests/utility/runconfig.json -O tests/utility
```

The results file produced after successfully running the script can be parsed with the function `load_results_utility` provided in `utils/analyse_results.py`.

## Visualizing results

A jupyter notebook to visualize and analyse the results is included at `notebooks/Analyse Results.ipynb`.