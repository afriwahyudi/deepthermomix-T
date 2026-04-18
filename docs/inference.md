# Documentation of 'inference' submodule

## Overview

This document provides comprehensive documentation for the `src\deepthermomix\inference` submodule, explaining the inference pipeline for predicting activity coefficients and thermodynamic properties across binary, ternary, and N-component mixtures. It serves as a reference for understanding how trained models generate predictions, compute uncertainties, and visualize phase equilibrium diagrams.

**Supported Features:**
- Binary mixture VLE phase equilibrium calculations via `binary.py` (P-x-y only)
- Gibbs energy analysis for binary system (N=2) and ternary system (N=3).
- Activity coefficient predictions for arbitrary N-component mixtures.

**Not Implemented:**
- Liquid-Liquid Equilibrium (LLE)
- Vapor-Liquid-Liquid Equilibrium (VLLE)
- Solid-Liquid Equilibrium (SLE)
- Phase equilibrium calculations for ternary and N-component mixtures

## Table of Contents

- [antoine_scrapper.py](#antoine_scrapperpy)
- [binary.py](#binarypy)
- [ternary.py](#ternarypy)
- [n_component.py](#n_componentpy)
- [utils.py](#utilspy)
- [Relations: Dependencies and Usage](#relations-dependencies-and-usage)

---

## antoine_scrapper.py

This module provides NIST WebBook integration for fetching Antoine equation parameters and saturation pressures required for vapor-liquid equilibrium (VLE) calculations.

### AntoineEquation Class

Manages caching and retrieval of Antoine parameters for saturation pressure predictions across temperature ranges.

#### Key Methods

**`__init__()`**

Initializes the scraper with:
- In-memory cache for Antoine parameters and component names
- HTTP session with browser-like headers to avoid NIST blocking
- Base URL for NIST WebBook API queries

**`_get_inchi(smiles: str) -> str`**

Converts SMILES string to InChI notation using RDKit for precise NIST compound identification.

**`_canonicalize(smiles: str) -> str`**

Standardizes SMILES to canonical form with stereochemistry preservation via RDKit.

**`_fetch_nist_params(smiles: str, name: Optional[str]) -> List[Dict]`**

Retrieves Antoine parameters from NIST WebBook with dual-search strategy:

1. **Primary**: InChI-based search (most precise)
2. **Fallback**: Chemical name search (if InChI fails)

Returns list of parameter sets with valid temperature ranges. Stores official NIST compound name for later use.

**`_query_nist(params: Dict) -> Tuple[List[Dict], Optional[str]]`**

Executes HTTP request to NIST WebBook and parses HTML response:

- Extracts compound H1 header for official name
- Locates Antoine parameter tables by `aria-label` or H3 context
- Parses temperature ranges, A, B, C coefficients from table cells
- Handles malformed data gracefully with fallback ranges

**Returns:** List of parameter dictionaries and extracted compound name.

**`get_Psat(smiles: str, T_kelvin: float, name: Optional[str]) -> float`**

Computes saturation pressure (bar) at given temperature:

$$P_{\text{sat}} = 10^{(A - \frac{B}{T + C})}$$

**Selection logic:**
1. **Priority 1**: Use parameter set where `T_min ≤ T_kelvin ≤ T_max` (interpolation)
2. **Priority 2**: Use closest range boundary (extrapolation with warning)

Returns fallback value of 1.0 bar if no parameters available.

**`get_stored_name(smiles: str) -> Optional[str]`**

Retrieves cached NIST compound name from previous scraping.

---

## binary.py

This module provides VLE prediction and visualization for binary (two-component) mixtures.

### VLEAnalyzer Class

Computes phase equilibrium diagrams and thermodynamic properties for binary systems.

#### Initialization

**`__init__(model, pipeline, device='cpu')`**

Sets up analyzer with:
- Trained model (in eval mode)
- Data pipeline for molecular graph construction
- CUDA/CPU device targeting
- Internal SMILES-to-name mapping from pipeline registry

#### Key Methods

**`_prepare_single_point(smiles_list: List[str], mole_fracs: List[float]) -> Data`**

Constructs a single PyTorch Geometric Data object for model inference:

- Converts SMILES to RDKit molecules
- Generates node features (21-dim vectors) and bond attributes (4-dim one-hot)
- Creates disjoint union graph with `mol_batch` offset tracking
- Stores composition as `component_mole_frac` tensor

**Returns:** Batched Data object ready for neural network input.

**`phase_calculation(smiles1: str, smiles2: str, T_kelvin: float, steps: int) -> pd.DataFrame`**

Generates complete binary VLE diagram at fixed temperature:

**Process:**
1. Sweep mole fraction x₁ from 0 to 1 in `steps` increments

2. For each composition:

    - Construct molecular graph

    - Predict activity coefficients ($\ln \gamma_1$, $\ln \gamma_2$)

    - Fetch saturation pressures ($P_{sat_{1}}$, $P_{sat_{2}}$) from NIST

    - Calculate partial pressures: $p_i = x_i \gamma_i P_{\text{sat},i}$

    - Compute total pressure: $P = \sum p_i$

    - Extract vapor mole fractions: $y_i = p_i / P$
    
3. Compute:

    - Excess Gibbs energy: $g^E/RT = \sum x_i \ln \gamma_i$

    - Mixing Gibbs energy: $\Delta_{\text{mix}}g/RT = g^E/RT + \sum x_i \ln x_i$


**Returns:** DataFrame with columns:
- `P`, `x1`, `x2`, `y1`, `y2` (phase equilibrium)
- `ln_gamma1`, `ln_gamma2`, `gamma1`, `gamma2` (activity coefficients)
- `g_excess_reduced`, `g_mix_reduced` (thermodynamic properties)

Metadata attributes store component names for plotting.

**`plot_vle(df: pd.DataFrame, title_prefix: Optional[str])`**

Generates publication-quality three-panel VLE diagram:

| Panel | Content | Axes |
|-------|---------|------|
| **P-x-y** | Bubble/dew curves | mol frac (x-axis) vs. pressure (y-axis) |
| **Activity Coefficients** | $\ln \gamma$ behavior across composition | mol frac vs. $\ln \gamma$ |
| **Gibbs Energy** | Excess and mixing contributions | mol frac vs. energy (RT units) |

**Formatting:**
- Times New Roman serif font, 14pt labels
- Zero-line reference for $\ln \gamma$ plot
- Legend and grid enabled
- Tight layout with 20×6 inch figure size

---

## ternary.py

This module provides VLE prediction and visualization for ternary (three-component) mixtures on simplex coordinates.

### TernaryAnalyzer Class

Computes energy surfaces for ternary systems using triangular (ternary) plots.

#### Initialization

**`__init__(model, pipeline, device='cpu')`**

Initializes analyzer with trained model, pipeline, and device configuration.

#### Key Methods

**`_prepare_batch(smiles_list: List[str], mole_fracs_batch: np.ndarray) -> Tuple[Tensors]`**

Prepares batched molecular graphs for multiple compositions:

- Constructs individual molecular graphs for each component
- Concatenates graphs with offset tracking
- Returns base tensors for repeated batching across compositions

**`_generate_simplex_grid(steps: int) -> np.ndarray`**

Generates equidistant grid points on ternary simplex satisfying $x_1 + x_2 + x_3 = 1$:

- Iterates over composition pairs with step resolution
- Enforces minimum fractions (1e-9) to avoid division-by-zero
- Removes duplicate normalized compositions
- Returns N × 3 array of mole fractions

**`predict_ternary_surface(smiles_list: List[str], steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]`**

Predicts activity coefficients and Gibbs energies across ternary composition space:

**Process:**
1. Generate simplex grid with `steps` resolution

2. Create disjoint union graphs with all components

3. Batch process compositions in chunks (128 per batch for memory efficiency)

4. For each chunk:

    - Construct graph tensors with proper node/edge offsetting

    - Forward pass through neural network → $\ln \gamma$ predictions

    - Compute excess Gibbs energy: $g^E/RT = \sum x_i \ln \gamma_i$

    - Compute mixing energy: $\Delta_{\text{mix}}g/RT = g^E/RT + \sum x_i \ln x_i$

5. Concatenate results across all chunks

**Returns:**
- `mole_fracs`: N × 3 grid of composition points
- `g_excess`: Activity-based contributions
- `g_mix`: Total mixing contributions
- `df`: DataFrame with all predictions and coordinates

**`plot(smiles_list: List[str], save_path: Optional[str]) -> pd.DataFrame`**

Generates side-by-side ternary contour plots:

**Left Panel**: Excess Gibbs energy ($g^E/RT$)
**Right Panel**: Mixing Gibbs energy ($\Delta_{\text{mix}}g/RT$)

**Features:**
- RdBu_r colormap (red for positive, blue for negative)
- 30-level contours with labeled grid lines
- Component names at simplex vertices with offset positioning
- Colorbar with energy units

Uses `mpltern` library for native ternary projections.

---

## n_component.py

This module provides activity coefficient predictions for N-component (binary, ternary, or higher) mixtures with composition sweeps.

### MultiComponentAnalyzer Class

Enables systematic scanning of composition space for arbitrary component counts.

#### Initialization

**`__init__(model, pipeline, device='cpu')`**

Sets up analyzer with model and pipeline. Caches SMILES-to-name mappings from pipeline registry and NIST scraper.

#### Key Methods

**`_get_name(smiles: str) -> str`**

Retrieves human-readable component name with priority order:

1. Pipeline registry (from `solvent_id_to_name`)

2. NIST WebBook cached result

3. SMILES string (fallback)

**`_prepare_batch_N(smiles_list: List[str], mole_fracs_batch: np.ndarray) -> Data`**

Constructs batched molecular graphs for N components across multiple compositions:

- Precomputes individual component graphs

- Repeats base graph structure for each composition in batch

- Applies node offset tracking to maintain disjoint union property

- Scales edge indices and batch assignments appropriately

Handles up to 128 compositions per batch for memory efficiency.

**`scan_composition(smiles_list: List[str], target_idx: int, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**

Performs composition sweep varying one component while maintaining equimolar distribution of others:

**Logic:**
- Target component varies from 0 to 1 in `steps` increments

- Remaining N-1 components share equal fractions of residual composition

- Normalizes fractions to ensure $\sum x_i = 1$

**Returns:**
- `x_axis`: Array of target component mole fractions

- `gamma`: Steps × N array of activity coefficients

- `ln_gamma`: Steps × N array of log-activity coefficients

**`plot_sweep(smiles_list: List[str], target_idx: int, steps: int, save_path: Optional[str], custom_names: Optional[List[str]])`**

Generates dual-panel activity coefficient plots:

| Panel | Content | Y-axis |
|-------|---------|--------|
| **Left** | $\gamma$ vs. composition sweep | Activity coefficient ($\gamma$) |
| **Right** | $\ln \gamma$ vs. composition sweep | Log-activity coefficient ($\ln \gamma$) |

**Features:**
- Separate line per component with legend
- Grid enabled with 0.3 alpha transparency
- Reference line at ln γ = 0
- Custom component names supported
- Optional PNG export at 300 DPI

---

## utils.py

This module provides inference evaluation utilities for model validation and uncertainty quantification.

### ComputeMetric Class

Computes comprehensive evaluation metrics and bootstrap confidence intervals.

#### Initialization

**`__init__(model, loader, device='cuda')`**

Sets up evaluator with:
- Trained model in eval mode
- PyG DataLoader for batched inference
- Loss functions: `MixMSELoss` (data-driven), `GibbsDuhemLoss` (physics-informed)

#### Key Methods

**`run_evaluation() -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]`**

Performs full-dataset evaluation with uncertainty tracking:

**Process:**
1. Iterate through all batches in loader
2. For each batch:
    - Forward pass → predictions + uncertainty (std dev)
    - Compute Gibbs-Duhem consistency loss (physics constraint)
3. Aggregate predictions and ground truth across batches
4. Compute global metrics

**Computed Metrics:**

| Metric | Equation | Usage |
|--------|----------|-------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum(\hat{y}_i - y_i)^2}$ | Primary regression error |
| **MAE**  | $\frac{1}{N}\sum\|\hat{y}_i - y_i\|$ | Robustness to outliers |
| **$R^2$**   | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | Explained variance |
| **GD Loss** | Physics constraint violation | Thermodynamic consistency |

**Returns:**
- `df_formatted`: String-aggregated results per system (delimiter-separated)
- `df_raw`: Detailed per-sample predictions with uncertainties
- Scalar metrics: `rmse_log`, `mae_log`, `r2_log`, `avg_gd_loss`

**Output DataFrame Columns:**
- `group_id`: Batch sample identifier
- `solv_i_id`: System/component ID
- `molefrac_i`: Mole fraction (delimiter-separated)
- `ln_gamma_exp`: Ground-truth activity coefficient
- `ln_gamma_pred`: Model prediction
- `uncertainty`: Predicted standard deviation
- `error_lin`: Linear-scale error

**`bootstrap_metrics(df_raw: pd.DataFrame, n_bootstrap: int) -> Dict`**

Computes 95% confidence intervals via non-parametric bootstrap resampling:

**Process:**
1. Resample dataset with replacement `n_bootstrap` times (default: 1000)
2. For each resample: compute RMSE, MAE, $R^2$
3. Calculate standard deviation of metric distributions

**Returns:** Dictionary with keys:
- `rmse_std`: Uncertainty in RMSE estimate
- `mae_std`: Uncertainty in MAE estimate
- `r2_std`: Uncertainty in $R^2$ estimate

**`_format_to_string(df_raw: pd.DataFrame) -> pd.DataFrame`**

Aggregates per-sample predictions to per-system format with delimiter-separated lists:

- Groups by `group_id` (system identifier)
- Joins all component properties with " / " delimiter
- Formats floats to appropriate precision (2 for fractions, 9 for predictions)

---

## Relations: Dependencies and Usage

### Dependency Structure

```
binary.py─────────┐
ternary.py────────┼──> antoine_scrapper.py
n_component.py────┘

utils.py ──> model/losses.py
```

### Internal Dependencies

- `binary.py`, `ternary.py`, `n_component.py` → `antoine_scrapper.py` (NIST parameter fetching)
- All analyzers depend on `pipeline._mol_to_graph()` (from data module)

### Used by Other Modules

- `development\scripts\main_scripts\`:

  - `evaluate_performance.py` → `utils.py`

  - `predict_binary.py` → `binary.py`

  - `predict_n_activity.py` → `n_component.py`

  - `predict_ternary.py` → `ternary.py`

*(Arrow indicates "imports" or "depends on")*

### Impact Summary

| Module | Impact | Notes |
|--------|--------|-------|
| `antoine_scrapper.py` | **High** | Saturation pressure accuracy affects VLE prediction quality across all analyzers. |
| `binary.py` | **High** | Binary VLE calculations are foundational for model validation. |
| `ternary.py` | **High** | Ternary surface predictions require precise activity coefficients and graph construction. |
| `n_component.py` | **Medium** | Composition sweep flexibility supports exploratory analysis and edge-case testing. |
| `utils.py` | **High** | Evaluation metrics and physics loss directly impact model quality assessment. |

> **Note:** The `inference` submodule depends on the `data` module for molecular graph construction and on the `model` module for loss functions. It has no internal circular dependencies.
> **Important:** `antoine_scrapper.py` is the only module in this submodule that requires internet connectivity, as it fetches Antoine equation parameters directly from the NIST WebBook API.

