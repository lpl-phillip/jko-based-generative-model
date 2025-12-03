# aggregation
## Files
# Operator-Learning JKO Generative Model

This repository contains code for a **generative model based on JKO / Wasserstein gradient flow**, connecting  
**operator learning** and **flow-based generative models**.

## Project Goal

- **Operator learning view:** Learn a mapping between *distributions* (or density functions), e.g. from ρₖ to ρₖ₊₁ along a JKO/Wasserstein gradient flow step.  
- **Flow-based generative view:** Learn a mapping from a simple base distribution (e.g. Gaussian) to a complex data distribution (e.g. images) so that we can **generate new samples**.  
- Our aim is to **bridge these two views**:  
  - Use a neural network to approximate the **JKO update operator** that pushes one distribution toward the minimizer of an energy (often involving a KL term).  
  - Reuse this learned operator (or its reverse flow) as a **data-efficient generative model**, especially in settings with limited training samples.

---

## File Overview

### `main_kl_Gaussiancombo_withMonitor.py`
Main training/experiment script.  
- Parses arguments, prepares data, builds the model, runs the JKO-style training loop with KL-based losses,  
  and handles logging/monitoring (plots, checkpoints, optional GIFs of trajectories).

### `arguments.py`
Command-line argument and hyperparameter definitions.  
- Central place to configure dataset type, JKO steps, batch sizes, learning rate, scheduler options,  
  monitoring flags, maximum training iterations, etc.

### `models.py`
Neural network architectures.  
- Contains convolutional blocks, encoder/decoder networks, and the main JKO/operator network used to map  
  from current data (or density representation) to its updated state.

### `dataset_utils.py`
Dataset preparation and sampling utilities.  
- Provides functions to create/load different datasets (e.g. Gaussian mixtures, toy dynamics, MNIST),  
  and returns them in a format suitable for the training loop in `main_kl_Gaussiancombo_withMonitor.py`.

### `shape_data.py`
Generators for 2D toy shape data.  
- Builds simple point-cloud–type datasets (e.g. rings, moons, stars, hearts, etc.) that are useful for  
  visualization and sanity checks of Wasserstein/JKO behavior in low dimensions.

### `loss_jko.py`
Loss functions and density-related utilities.  
- Implements KL-related terms and Gaussian-mixture log densities, and may include additional discrepancy  
  measures (such as kernel/MMD-type losses) for comparing model distributions to target distributions.

### `utils.py`
General helper functions.  
- Utilities for logging and saving results, PCA projection and visualization, gradient/NaN checks,  
  writing loss vectors to files, and other small tools that support the main training script.

### `test.py`
Small test / debugging script.  
- Used to run simplified experiments or sanity-check individual components (data loading, model blocks,  
  or loss terms) before launching full training runs.



for the main branch
- `main_aggregation_withMonitor.py`: main script for training & monitoring.
- `models.py`: core model definitions (transformer).
- `GNN.py`: GNN-based model variant (doesn't work very well).
- `dataset_utils.py`: data loading and preprocessing.
- `shape_data.py`: generate aggregation shapes / initial data.
- `loss_jko.py`: JKO loss and related functions.
- `utils.py`: helper functions.
- `arguments.py`: command-line argument parser.

