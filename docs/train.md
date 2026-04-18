# Documentation of 'train' submodule

## Overview

This document provides comprehensive documentation for the `src\deepthermomix\train` submodule, explaining the training pipeline, loss computation, and hyperparameter optimization integration.

## Table of Contents

- [trainer.py](#trainerpy)
- [Relations: Dependencies and Usage](#relations-dependencies-and-usage)

---

## trainer.py

### DTMPNNTrainer

A trainer class define the complete training workflow, including model optimization, validation, early stopping, and checkpoint management with integrated support for physics-informed constraints via Gibbs-Duhem loss.

**Key Functionality:**

- **Dual Loss Training:** Combines data-driven loss (MixedMSELoss) with physics constraints (GibbsDuhemLoss) for thermodynamically consistent predictions.
- **Early Stopping:** Patience-based mechanism to halt training when validation loss plateaus, preventing overfitting.
- **Checkpoint Management:** Automatic persistence of model weights, optimizer state, and training history.
- **Optuna Integration:** Support for hyperparameter optimization trials with trial pruning.
- **Comprehensive Logging:** Detailed epoch-by-epoch metrics for training, validation, and test sets.

**Core Methods:**

| Method | Purpose |
|--------|---------|
| `compute_losses()` | Calculates total loss as weighted sum of data-driven and Gibbs-Duhem components. |
| `train_epoch()` | Executes one training epoch: forward pass, backpropagation, weight updates over entire training set. |
| `validate()` | Evaluates model on validation or test loader without gradient computation. |
| `train()` | Main training loop with early stopping, checkpoint saving, and optional trial pruning. |
| `save_checkpoint()` | Persists model state dict, optimizer state, and training history to disk. |
| `load_checkpoint()` | Restores model, optimizer, and history from a saved checkpoint. |
| `plot_history()` | Visualizes training curves (total loss, data-driven loss, Gibbs-Duhem loss). |

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DTMPNN` | — | Neural network model to train. |
| `train_loader` | DataLoader | — | Training set loader. |
| `val_loader` | DataLoader | — | Validation set loader. |
| `test_loader` | DataLoader | — | Test set loader. |
| `include_gd` | bool | — | Enable/disable Gibbs-Duhem constraint loss. |
| `device` | str | — | Compute device (`'cuda'` or `'cpu'`). |
| `lr` | float | `1e-6` | Learning rate for optimizer. |
| `weight_decay` | float | `1e-5` | L2 regularization coefficient. |
| `data_driven_weight` | float | — | Weighting factor for data-driven loss component. |
| `gd_weight` | float | — | Weighting factor for Gibbs-Duhem constraint loss. |
| `constraint_type` | str | — | Constraint classification identifier (`'none'`, `'soft'`, or `'hard'`). |

---

## Relations: Dependencies and Usage

### Dependency Structure

```
trainer.py → model\losses.py
```

### Internal Dependencies

None

### Used by Other Modules

- `development\scripts\main_scripts\`:
    - `execute_hpo.py`
    - `execute_training.py`

*(Arrow indicates "imports" or "depends on")*

### Impact Summary

| Module | Impact | Notes |
|--------|--------|-------|
| `trainer.py` | **Medium** | Changes only affect training scripts; prediction workflows unaffected. |

> **Note:** The `train` submodule depends on `model\losses.py` and must be updated if loss computation changes. However, it has no dependencies on other `deepthermomix` submodules and does not affect inference or ensemble prediction.

