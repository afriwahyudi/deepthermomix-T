# Documentation of 'data' submodule

## Overview

This document provides comprehensive documentation for the `src\deepthermomix\data` submodule, explaining the data pipeline architecture, molecular graph construction, and dataset splitting strategies. It serves as a reference for understanding how raw mixture data is processed into model-ready graph representations and how datasets are partitioned for training, validation, and testing.

## Table of Contents

- [data_pipeline.py](#data_pipelinepy)
- [datasplit_scheme.py](#datasplit_schemepy)
- [Relations: Dependencies and Usage](#relations-dependencies-and-usage)

---

## data_pipeline.py

This file defines the complete data pipeline for processing N-component mixture data into canonicalized form and PyTorch Geometric graph representations suitable for neural network training.

### Pipeline Overview

The `DataPipeline` class manages two essential input sources:

1. **Component Registry CSV** (e.g., `components.csv`)
    - Columns: `solvent_name`, `solvent_id`, `smiles_can`
    - Maintains canonical SMILES strings for all available chemical components
    - Extensible: new components added via their SMILES string

2. **Raw VLE Data CSV** (e.g., experimental mixture data)
    - Columns: `solv_i_id`, `molefrac_i`, `ln_gamma_i`
    - Stores component IDs, mole fractions, and ground-truth activity coefficients
    - Delimiter-separated lists per row (e.g., "solvent_587 / solvent_604")

### DataPipeline Class

#### Key Methods

**`load_components(filepath: str) -> pd.DataFrame`**

Loads and validates the component registry. Creates two mapping dictionaries:
- `solvent_id_to_smiles`: Maps component ID → canonical SMILES
- `solvent_id_to_name`: Maps component ID → chemical name

**`parse_systems(raw_data_df: pd.DataFrame) -> pd.DataFrame`**

Extracts component lists from the `solv_i_id` column by splitting on "/" delimiter and stripping whitespace. Returns a new DataFrame with `component_list` column.

**`parse_numlist(system_parsed_df: pd.DataFrame) -> pd.DataFrame`**

Parses numerical properties (mole fractions and activity coefficients) from string format into lists. Populates:
- `molefrac_list`: List of float mole fractions
- `ln_gamma_list`: List of float activity coefficients

**`assign_system_ids(df: pd.DataFrame) -> pd.DataFrame`**

Generates permutation-invariant System IDs based on the **sorted set** of components while preserving input data order.

**Logic:**
- Row 1: `['Water', 'Ethanol']` → sorted set: `{'Ethanol', 'Water'}` → System ID: 0
- Row 2: `['Ethanol', 'Water']` → sorted set: `{'Ethanol', 'Water'}` → System ID: 0 (same)

This ensures different orderings of the same mixture composition receive identical System IDs, enabling proper stratification during train/val/test splitting.

### Molecular Graph Construction

**`construct_graphs(canonical_df: pd.DataFrame) -> List[Data]`**

Converts processed mixture data into PyTorch Geometric `Data` objects. Key features:

- **Disjoint Union Graphs**: Multiple molecules per system are combined into a single graph via node offset tracking
- **Caching**: Molecular structures (RDKit molecules and graphs) are cached to avoid redundant computation
- **Diagnostic Reporting**: Tracks skipped systems due to missing SMILES, invalid molecules, or no valid components
- **System Classification**: Automatically labels mixtures as "Binary mixture", "Ternary mixture", or "N components mixture"

**Graph Attributes (per Data object):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor [N_atoms, 21] | Node features (see below) |
| `edge_index` | torch.Tensor [2, N_edges] | Undirected edge connectivity |
| `edge_attr` | torch.Tensor [N_edges, 4] | Bond type one-hot vectors |
| `mol_batch` | torch.Tensor [N_atoms] | Component index for each atom |
| `component_batch` | torch.Tensor [N_components] | Unique component indices |
| `component_names` | list[str] | Component IDs (e.g., "solvent_587") |
| `actual_names` | list[str] | Full chemical names |
| `component_mole_frac` | torch.Tensor [N_components] | Mole fractions |
| `component_ln_gammas` | torch.Tensor [N_components] | Ground-truth activity coefficients |
| `system_type` | str | Mixture classification |
| `system_id` | int | Permutation-invariant system identifier |

### Atomic Features (Node Features)

Each atom is represented by a 21-dimensional vector combining chemical properties:

| Feature | Size | Type | Description |
|---------|------|------|-------------|
| Atom type (one-hot) | 11 | int | {H, C, N, O, F, Si, P, S, Cl, Br, I} |
| Hybridization (one-hot) | 3 | int | {sp³, sp², sp} |
| Aromaticity | 1 | int | Binary flag |
| Ring membership | 1 | int | Binary flag |
| H-bond donor | 1 | int | Binary flag |
| H-bond acceptor | 1 | int | Binary flag |
| Formal charge | 1 | int | Integer value |
| Partial charge (Gasteiger) | 1 | float | Computed via RDKit |
| Atomic mass | 1 | float | Normalized (×0.01) |
| Van der Waals radius | 1 | float | From periodic table |
| Degree | 1 | int | Total connectivity |

**Total: 21 features per atom**

### Bond Features (Edge Attributes)

Each bond is encoded as a 4-dimensional one-hot vector:

$$\text{bondvec} = [b_{\text{single}}, b_{\text{double}}, b_{\text{triple}}, b_{\text{aromatic}}]$$

Edges are undirected: both (i→j) and (j→i) are added with identical attributes.

**`_mol_to_graph(mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**

Internal utility converting RDKit molecules to node features, edge indices, and bond attributes. Automatically adds explicit hydrogens and computes Gasteiger partial charges.

**`run_pipeline(raw_csv: str, verbose: bool) -> Tuple[pd.DataFrame, List[Data]]`**

Defines the complete pipeline with five steps:
1. Load raw CSV data
2. Parse system composition strings
3. Parse numerical lists
4. Assign permutation-invariant System IDs
5. Construct molecular graphs

**Returns:** Processed DataFrame and list of PyG Data objects ready for training.

---

## datasplit_scheme.py

This module provides system-disjoint splitting logic to ensure no mixture system appears in multiple train/val/test partitions, preventing data leakage.

### Effective Component Count

**`get_effective_n(n_components: int, fractions: Union[List, np.ndarray, torch.Tensor], tol: float) -> int`**

Determines the effective number of components by accounting for infinite dilution/pure component scenarios.

**Logic:**
- If `max(fractions) >= 1.0 - tol`: treat as N=1 (single-component or pure limit)
- Otherwise: return nominal N

This enables proper stratification of binary, ternary, etc., systems across train/val/test splits.

### System-Disjoint Splitting

**`system_disjoint_split(data, test_size, val_size, random_state, stratify_by_components, dilution_tol)`**

Splits data at the system level (not row level) to maintain composition diversity across partitions.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | list[Data] or pd.DataFrame | — | PyG Data objects or DataFrame |
| `test_size` | float | 0.15 | Fraction of systems for test set |
| `val_size` | float | 0.15 | Fraction of systems for validation |
| `random_state` | int | 42 | Random seed for reproducibility |
| `stratify_by_components` | bool | False | Maintain system-type proportions |
| `dilution_tol` | float | 1e-4 | Tolerance for effective N determination |

**Returns:** Tuple of `(train_data, val_data, test_data)`

**Workflow:**

1. **Identify Unique Systems**: Extract all unique `system_id` values from input
2. **Compute Component Counts** (if stratifying): Determine effective N for each system
3. **Stratified Split** (if enabled): Use `sklearn.model_selection.train_test_split` with `stratify` parameter to maintain N-ary distribution
4. **Assign Data to Partitions**: Route each sample to train/val/test based on its system_id
5. **Report Statistics**: Print distribution of effective components and sample counts per partition

**Example Output:**
```
Effective Component Distribution (N=1 includes x->1):
  Train: {1: 45, 2: 120, 3: 78}
  Val:   {1: 8, 2: 25, 3: 15}
  Test:  {1: 7, 2: 20, 3: 12}

Datapoints -> Train: 243, Val: 48, Test: 39
Unique systems -> Train: 95, Val: 20, Test: 19
```

**Input Flexibility:**

The function handles both:
- **PyG Data lists**: Extracts `system_id`, component counts, and mole fractions from graph attributes
- **Pandas DataFrames**: Uses dedicated columns (`system_id`, `component_names`, `component_mole_frac`, etc.)

---

## Relations: Dependencies and Usage

### Dependency Structure

```
(No internal submodule dependencies)
```

### Internal Dependencies

None (does not depend on other `deepthermomix` submodules)

### Used by Other Modules

- `train\trainer.py` → `data_pipeline.py`, `datasplit_scheme.py`
- `development\scripts\main_scripts\`:
  - `evaluate_performance.py` → `data_pipeline.py`, `datasplit_scheme.py`
  - `execute_hpo.py` → `data_pipeline.py`, `datasplit_scheme.py`
  - `execute_training.py` → `data_pipeline.py`, `datasplit_scheme.py`
  - `predict_binary.py` → `data_pipeline.py`
  - `predict_n_activity.py` → `data_pipeline.py`
  - `predict_ternary.py` → `data_pipeline.py`

*(Arrow indicates "imports" or "depends on")*

### Impact Summary

| Module | Impact | Notes |
|--------|--------|-------|
| `data_pipeline.py` | **High** | Changes affect graph construction and all downstream training/inference. |
| `datasplit_scheme.py` | **High** | Changes alter train/val/test composition and stratification logic. |

> **Note:** The `data` submodule is foundational—it has no dependencies on other `deepthermomix` submodules. All upstream modules depend on its outputs (graph representations and data splits).