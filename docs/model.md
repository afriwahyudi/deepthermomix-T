# Documentation of 'model' submodule

## Overview

This document provides comprehensive documentation for the `src\deepthermomix\model` submodule, explaining the architecture and functionality of components whose implementation details are condensed in the source code. It serves as a reference for understanding the neural network design, thermodynamic constraints, ensemble methods, and loss functions implemented.

## Table of Contents

- [architecture.py](#architecturepy)
- [ensemble_wrapper.py](#ensemble_wrapperpy)
- [losses.py](#lossespy)
- [model_loader.py](#model_loaderpy)
- [Relations: Dependencies and Usage](#relations-dependencies-and-usage)

---

## architecture.py

This file defines the neural network architecture used to predict activity coefficients ($\ln \gamma_i$) from molecular structures and mixture compositions. It includes the graph encoder, the mixing interaction layer, and the top-level model that enforces thermodynamic constraints.

### MPNNLayer & MPNNBlock

The **Message Passing Neural Network (MPNN)** serves as the graph encoder, processing molecular graphs to generate fixed-size embeddings for each chemical component.

**Structure:**
- **MPNNLayer:** A single message-passing iteration where edge features and neighbor node features are aggregated to update node states.
- **MPNNBlock:** A stack of `MPNNLayer` instances with residual connections to facilitate gradient flow in deeper networks.
- **Pooling:** Node embeddings are aggregated via `global_mean_pool` to produce a single vector representation (`comp_emb`) for each distinct molecule in the batch.

### DeepThermoMix

This module models the non-ideal interactions between components in a mixture, acting as a learned "mixing rule."

**Functionality:**
Computes a latent representation of the component within the mixture context by combining:
1. **Component Embedding:** Structural information from the MPNN.
2. **Mole Fraction:** The concentration of the component ($x_i$).
3. **Mixture Context:** An aggregated context vector representing the global mixture state.

This layer is used twice in the "hard" constraint formulation: once for the actual mixture state and once for the ideal (pure) state.

### DTMPNN

The main model class integrating the graph encoder and mixing layer. It supports three constraint modes:

| Mode | Behavior |
|------|----------|
| `'none'` | Directly predicts $\ln \gamma_i$ without thermodynamic enforcement. |
| `'soft'` | Directly predicts $\ln \gamma_i$ with regularization via `GibbsDuhemLoss`. |
| `'hard'` | Learns a pseudo-energy function; derives thermodynamically consistent coefficients via automatic differentiation. |

**Theoretical Basis:**

$$\ln \gamma_i = \left(\frac{g^E}{RT}\right) + \frac{\partial}{\partial x_i} \left(\frac{g^E}{RT}\right) - \sum_{j}^N x_j \frac{\partial}{\partial x_j} \left(\frac{g^E}{RT}\right)$$

**Implementation Details:**

The model calculates the difference between two states using shared weights:

$$\frac{g^E}{RT} = \sum_i^N x_i \cdot \left( \Psi(\mathcal{h}_{mix,i}) - \Psi(\mathcal{h}_{pure,i}) \right)$$

where $\Psi$ is the learned potential from `output_layer_hard`. The gradient is computed via `torch.autograd()`.

The final prediction assembles three terms:
- **Term A:** Total system energy $\left({g^E}/{RT}\right)$

- **Term B:** Direct gradient $\frac{\partial}{\partial x_i} \left(\frac{g^E}{RT}\right)$

- **Term C:** Correction term $\sum_{j}^N x_j \frac{\partial}{\partial x_j} \left(\frac{g^E}{RT}\right)$

$$\gamma^{pred}_i = \text{Term A} + \text{Term B} - \text{Term C}$$

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `node_dim` | int | Dimension of input node features. |
| `edge_dim` | int | Dimension of input edge features. |
| `graph_hidden_dim` | int | Hidden dimension size for the MPNN. |
| `latent_dim` | int | Hidden dimension for the mixing layer. |
| `constraint_type` | str | `'none'`, `'soft'`, or `'hard'`. |

**Returns:**
- `prediction`: Tensor [N_total_components] of predicted $\ln \gamma$
- `mixture_latent_vectors`: Latent representations for analysis/visualization
- `comp_emb`: Structural embeddings of components

---

## ensemble_wrapper.py

This file provides utilities for model ensembling, enabling uncertainty quantification via prediction variance across multiple trained `DTMPNN` instances.

### EnsembleWrapper

A PyTorch module wrapping a list of trained models and aggregating their outputs.

**Forward Method:**
- Executes all constituent models on input data and computes mean prediction
- Extracts the first element if models return tuples
- **Returns:** `mean_output`, `None`, `None`

**get_uncertainty:**
Computes statistical moments of ensemble predictions.

| Return Value | Formula | Description |
|--------------|---------|-------------|
| `mean_output` | $\mu = \frac{1}{N}\sum \ln \gamma_i$ | Average prediction |
| `std_output` | $\sigma = \sqrt{\frac{1}{N}\sum (\ln \gamma_i - \mu)^2}$ | Epistemic uncertainty |

### load_ensemble

Instantiates an `EnsembleWrapper` from a directory of model checkpoints.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | str | — | Path to directory containing `.pt` files. |
| `constraint_type` | str | `'hard'` | Physical constraint mode. |
| `device` | str | — | Compute device (`'cpu'` or `'cuda'`). |

**Returns:** Initialized `EnsembleWrapper` instance in `eval()` mode.

**Behavior:**
1. Scans for `*.pt` files and sorts deterministically
2. Calls `model_loader.load_model` for each file
3. Raises `ValueError` if directory is empty

---

## losses.py

Defines loss functions operating on mixture-structured data, evaluated at the mixture level using component-wise predictions and batch indices.

### MixedMSELoss

Computes loss by calculating Sum of Squared Errors (SSE) per mixture, then taking the mean—suitable for fair comparison across mixtures with varying component counts.

**Formula:**

$$\mathcal{L}_{data} = \frac{1}{M} \sum_{m=1}^{M} \sum_{i=1}^{N_m} \left( y_{m,i}^{\text{pred}} - y_{m,i}^{\text{true}} \right)^2$$

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_pred` | torch.Tensor | Predictions [N_total_components] |
| `batched_data` | torch_geometric.data.Batch | Contains `component_ln_gammas` and `component_batch_batch` |

**Returns:** Scalar-valued loss

### GibbsDuhemLoss

Penalizes violations of Gibbs–Duhem consistency across components within each mixture.

**Theoretical Basis:**

The excess Gibbs energy is:

$$\mathrm{g^E} = RT \sum_{i}^N x_i \ln\gamma_i$$

At constant T and P, Gibbs-Duhem demands:

$$\sum_{i} x_i d\ln \gamma_i = 0$$

**Key Insight:**

$$v_j = \sum_{i} x_i \frac{\partial \ln \gamma_i}{\partial x_j} = c$$

Each term must be constant. Therefore, enforcement requires **variance minimization** rather than residual summation (refer to the original paper for the derivation).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_type` | str | `'optimized'` | `'explicit'` --> $O(N^2$) or `'optimized'` --> $O(N)$ |
| `data` | PyG Data | — | Contains `component_mole_frac` and `component_batch` |
| `ln_gamma_calc` | torch.Tensor | — | Model outputs `[num_components_total]` |

**Returns:** Scalar Gibbs-Duhem residual

---

## model_loader.py

Provides utilities for loading trained `DTMPNN` models with robust hyperparameter inference from saved weights.

### infer_model_architecture

Reverse-engineers architectural hyperparameters from a PyTorch state dictionary.

**Deduced Parameters:**
- `node_dim`, `graph_hidden_dim`: From `graph_block.layers.0.lin_node.weight`
- `edge_dim`: From `graph_block.layers.0.lin_edge.weight`
- `context_dim`, `latent_dim`: From interaction and gate MLP weights
- `graph_layers`: From maximum layer index in `graph_block.layers`

**Returns:** Dictionary of arguments for `DTMPNN` instantiation

### load_model

Primary function for loading model checkpoints with format flexibility and safety.

**Workflow:**
1. Registers numpy scalars as safe globals (prevent pickling errors)
2. Loads checkpoint (handles `model_state_dict`, `state_dict`, or raw state)
3. Infers architecture via `infer_model_architecture`
4. Handles constraint types (default: `'soft'`)
5. Instantiates and loads weights (fallback to `strict=False`)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_path` | str | — | Path to `.pt` or `.pth` file. |
| `constraint_type` | str | `'soft'` | Physical constraint mode. |
| `verbose` | bool | False | Print inferred parameters and status. |

**Returns:** Loaded `DTMPNN` instance in `eval()` mode

---

## Relations: Dependencies and Usage

### Dependency Structure

```
ensemble_wrapper.py → model_loader.py → architecture.py
```

### Internal Dependencies
- `ensemble_wrapper.py` → `model_loader.py` → `architecture.py`


### Used by Other Modules
- `inference\utils.py` → `losses.py`
- `train\trainer.py` → `losses.py`
- `development\scripts\main_scripts\`:
    - `evaluate_performance.py` → `ensemble_wrapper.py`
    - `execute_hpo.py` → `architecture.py`
    - `execute_training.py` → `architecture.py`
    - `predict_binary.py` → `ensemble_wrapper.py`
    - `predict_n_activity.py` → `ensemble_wrapper.py`
    - `predict_ternary.py` → `ensemble_wrapper.py`

*(Arrow indicates "imports" or "depends on")*

### Impact Summary

| Module | Impact | Notes |
|--------|--------|-------|
| `architecture.py` | **High** | Changes affect all downstream dependencies. |
| `losses.py` | **High** | Changes break training and inference validation. |
| `ensemble_wrapper.py` | **Medium** | Changes only affect prediction scripts. |
| `model_loader.py` | **Medium** | Changes affect ensemble loading. |

> **Note:** The `model` submodule is foundational—it has no dependencies on other `deepthermomix` submodules.

