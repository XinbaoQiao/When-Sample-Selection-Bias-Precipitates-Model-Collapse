# Quick Start Guide

## Environment Configuration

```bash
# Create environment
conda env create -f environment.yml
conda activate synthetic_data

# Or use pip
pip install -r requirements.txt
```

Core Dependencies: `torch`, `torchvision`, `transformers`, `diffusers`, `geomloss`, `accelerate`, `pot`.

## Usage

### 1. Running Sub-experiments

**Gaussian Modeling Analysis**

```bash
python subexperiments/Gaussian_Modeling/gaussian_correct.py
```

**Computation Overhead Benchmark**

```bash
python subexperiments/Computation_Overhead/benchmark_cifar10_time.py
python subexperiments/Computation_Overhead/plot_time_benchmark.py
```

**Barycenter Convergence**

```bash
python subexperiments/Barycenter_Convergence/wasserstein_barycenter_convergence_experiment.py
```

**Calibrated Gradient Analysis**

```bash
python subexperiments/Calibrated_Gradient/ot_distance_analysis_single.py
```


### 2. Run Biased Verification Experiment

```bash
python main.py --experiment_type biased_verification --dataset cifar10 --data_strategy accumulate_subsample
```

### 3. Run GEM (Our Methods) Experiment

```bash
# Scheme I: Local Greedy
python main.py --experiment_type gem --dataset cifar10 --num_clients 5 --selection_method gem --gem_method local_greedy

# Scheme II: Wasserstein Barycenter
python main.py --experiment_type gem --dataset cifar10 --num_clients 5 --selection_method gem --gem_method barycenter
```

### 4. Using run.sh Script for all baselines

The `run.sh` script facilitates running multiple biased verification experiments sequentially or in parallel.

```bash
# Run experiments sequentially (Default)
./run.sh

# Run experiments in parallel
./run.sh parallel
```
