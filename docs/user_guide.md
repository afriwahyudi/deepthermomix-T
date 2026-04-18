# User Guide

## Overview

The following ML models protocols are available for making predictions:

- **Protocol I**: Trained on $N_{\text{eff}}=2$ systems
- **Protocol II**: Trained on $N_{\text{eff}}=3$ systems
- **Protocol III**: Trained on $N_{\text{eff}}=\{2,3\}$ systems
- **Protocol IV**: Trained on $N_{\text{eff}}=\{1,2,3\}$ systems

## Quick Start: Prediction Tools

### Binary VLE Predictions

Use one of the following approaches:
- Automated scripts: `predict/easy_use/binary_vle/`
- Script: `predict/main_script/predict_binary.py` for single processing

### Ternary System Predictions

Use one of the following approaches:
- Automated scripts: `predict/easy_use/ternary_surface/`
- Script: `predict/main_script/predict_ternary.py` for single processing

### N-Component Activity Predictions

Use one of the following approaches:
- Automated scripts: `predict/easy_use/arbitrary_n/`
- Script: `predict/main_script/predict_n_activity.py` for single processing

## Analysis and Results

### Vapor-Liquid Equilibrium for Binary Systems

Generate VLE data using the prediction tools above. Analyze results in `notebooks/compare_system.ipynb`.

### Gibbs Energy Analysis

Analyze excess Gibbs energy and activity coefficients using the scripts and protocols provided in the prediction modules.

## Reference: AspenPlus Compound Mappings

Varying compounds in `\application\aspen\` via COM automation requires specific AspenPlus database conventions. The following table lists common compounds and their mappings:

| Compound Name | Aspen Convention | SMILES |
|---|---|---|
| 1-butanol | `N-BUTANOL` | `CCCCO` |
| 1-pentanol | `1-PENTANOL` | `CCCCCO` |
| 1-propanol | `1-PROPANOL` | `CCCO` |
| Acetic acid | `ACETIC-ACID` | `CC(=O)O` |
| Acetone | `ACETONE` | `CC(=O)C` |
| Acetonitrile | `ACETONITRILE` | `CC#N` |
| Aniline | `ANILINE` | `C1=CC=C(C=C1)N` |
| Benzene | `BENZENE` | `C1=CC=CC=C1` |
| Carbon disulfide | `CARBON-DISULFIDE` | `C(=S)=S` |
| Chloroform | `CHLOROFORM` | `ClC(Cl)Cl` |
| Cyclohexane | `CYCLOHEXANE` | `C1CCCCC1` |
| Diethanolamine | `DIETHANOLAMINE` | `NCCO` |
| Diethylamine | `DIETHYL-AMINE` | `CCNCC` |
| Dimethyl sulfoxide | `DIMETHYL-SULFOXIDE` | `CS(=O)C` |
| Ethanethiol | `ETHYL-MERCAPTAN` | `CCS` |
| Ethanol | `ETHANOL` | `CCO` |
| Ethylamine | `ETHYL-AMINE` | `CCN` |
| Formic acid | `FORMIC-ACID` | `O=CO` |
| Furfural | `FURFURAL` | `C1=COC(=C1)C=O` |
| Glycerol | `GLYCEROL` | `C(C(CO)O)O` |
| Hexafluorobenzene | `PERFLUOROBENZENE` | `C1(=C(C(=C(C(=C1F)F)F)F)F)F` |
| Methanol | `METHANOL` | `CO` |
| Methylcyclopentane | `METHYLCYCLOPENTANE` | `CC1CCCC1` |
| Monoethanolamine | `MONOETHANOLAMINE` | `NCCO` |
| n-Butylamine | `N-BUTYL-AMINE` | `CCCCN` |
| N-methylpyrrolidone | `N-METHYL-2-PYRROLIDONE` | `CN1CCCC1=O` |
| Nitrobenzene | `NITROBENZENE` | `C1=CC=C(C=C1)[N+](=O)[O-]` |
| Perfluorohexane | `PERFLUORO-N-HEXANE` | `C(C(C(C(F)(F)F)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F` |
| Phenol | `PHENOL` | `C1=CC=C(C=C1)O` |
| Propionic acid | `PROPIONIC-ACID` | `CCC(=O)O` |
| Pyridine | `PYRIDINE` | `C1=CC=NC=C1` |
| Thiophene | `THIOPHENE` | `C1=CSC=C1` |
| Toluene | `TOLUENE` | `CC1=CC=CC=C1` |
| Triethylamine | `TRIETHYL-AMINE` | `CCN(CC)CC` |
| Water | `WATER` | `O` |

### Example Binary Test Systems

- **Positive deviation**: Ethanol + Toluene; Methanol + Cyclohexane
- **Negative deviation**: Acetic acid + Ethanol; Phenol + Aniline
- **Near-ideal**: Benzene + Toluene; Methanol + Ethanol
- **Azeotropic**: Ethanol + Water; Chloroform + Acetone
- **Immiscible**: Water + Toluene; Water + Cyclohexane
- **Special cases**: Perfluorohexane + Hexafluorobenzene; Glycerol + Water

---

## Notebook Usage Tutorials

### ML Model Predictions

#### Binary VLE Prediction

Generate a VLE isotherm for an ethanol-water system at 298.15 K:

```python
import torch
from deepthermomix.inference.binary import VLEAnalyzer
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = DataPipeline('development/datasets/components.csv')
model = load_ensemble('path/to/ensemble', constraint_type='hard', device=device)
analyzer = VLEAnalyzer(model, pipeline)

df_vle = analyzer.phase_calculation(
    'CCO',      # Ethanol
    'O',        # Water
    T_kelvin=298.15,
    steps=100
)

analyzer.plot_vle(df_vle)
df_vle.to_csv('vle_results.csv', index=False)
```

#### Ternary System Surface

Visualize Gibbs energy across a ternary mixture composition space:

```python
import torch
from deepthermomix.inference.ternary import TernaryAnalyzer
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = DataPipeline('development/datasets/components.csv')
model = load_ensemble('path/to/ensemble.pt', constraint_type='hard', device=device)
analyzer = TernaryAnalyzer(model, pipeline, device=device)

df = analyzer.plot(
    ['CCO', 'C1=CC=CC=C1', 'C1CCCCC1'],  # Ethanol, Benzene, Cyclohexane
    save_path='ternary_surface.png'
)

df.to_csv('ternary_results.csv', index=False)
```

#### Multi-Component Composition Sweep

Examine how activity coefficients change as one component varies in a multi-component mixture:

```python
import torch
from deepthermomix.inference.n_component import MultiComponentAnalyzer
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = DataPipeline('development/datasets/components.csv')
model = load_ensemble('path/to/ensemble.pt', constraint_type='hard', device=device)
analyzer = MultiComponentAnalyzer(model, pipeline, device=device)

analyzer.plot_sweep(
    ['CCO', 'C1=CC=CC=C1', 'C1CCCCC1'],  # Ethanol, Benzene, Cyclohexane
    target_idx=0,
    steps=50,
    save_path='n_component_sweep.png'
)
```

### AspenPlus COM Simulations

#### Binary System Simulations

Execute `aspen_binary.py` or use Jupyter notebook:

```python
from application.aspen.aspen_binary import AspenVLE

vle = AspenVLE()
df = vle.get_VLE_from_aspen(
    ['ETHANOL', 'WATER'],
    npoint=100,
    temperature=298.15,
    model_name='NRTL')
```

Supported activity models: `COSMOSAC`, `NRTL`, `UNIFAC`, `UNIQUAC`, `WILSON`. Results include pressure, composition, activity coefficients, and Gibbs energy properties. You can add other activity models by making `.bkp` files yourself.

#### Ternary System Simulations

Execute `aspen_ternary.py` or use Jupyter notebook:

```python
from application.aspen.aspen_ternary import AspenTernary

ternary = AspenTernary()
df = ternary.get_ternary_data(
    ['ETHANOL', 'BENZENE', 'CYCLOHEXANE'],
    bkp_path='ternary_analysis/tern_nrtl.bkp',
    steps=60,
    temperature=298.15)
```

Output includes activity coefficients and Gibbs energy of mixing across the composition space.

#### Process Management

Both scripts implement robust Aspen process management:
- Automatic cleanup of persistent Aspen processes via `kill_all_aspen_processes()`
- Periodic COM object renewal during long simulation batches
- Garbage collection after intensive operations
- Error recovery with automatic Aspen restart

> **Note**: 
> - Requires licensed AspenPlus (V15 or later)
> - Ensure convention strings match your local database
> - **For NRTL, UNIQUAC, WILSON**: Can be fragile. Set `aspen.Visible = 1` to debug hangs. Usually caused by recalculation of binary interaction parameters after injecting components
> - **For COSMOSAC and UNIFAC**: Generally robust and reliable

### Comparison Notebook

The `notebooks/compare_system.ipynb` notebook provides comprehensive visualization tools for comparing predictions across classical and ML models.

**Prerequisites**: Ensure `.csv` files exist in `outputs/aspen` and `outputs/inference`

#### Setup

```python
import os
os.chdir("../..")
```

#### Main Configuration

Control output with the `PLOT_SCOPE` variable:
- `'ALL'`: Classical (ASPEN) and ML models together
- `'ASPEN'`: Classical models only (WILSON, NRTL, UNIQUAC, UNIFAC, COSMOSAC)
- `'ML'`: ML models only (Protocol I-IV)

#### Section 1: Binary System Comparison

**Purpose**: Visualize P-x-y diagrams with thermodynamic properties.

```python
PLOT_SCOPE = 'ALL'
aspen_sys_files = ('TOLUENE', 'WATER')
ml_sys_files = ('Toluene', 'WATER')
display_names = ('TOLUENE', 'WATER')
all_aspen_models = ['WILSON', 'NRTL', 'UNIFAC']
all_ml_models = ['protocol_III']
```

**Output**: 2×2 grid showing excess Gibbs energy, activity coefficients, mixing Gibbs energy, and P-x-y diagram.

#### Section 2: Multiple Systems Comparison

**Purpose**: Compare models across multiple binary systems.

```python
aspen_systems = [('ETHANOL', 'WATER'),
                 ('N-BUTANOL', 'WATER'),
                 ('TOLUENE', 'WATER')]
ml_systems = [('ETHANOL', 'WATER'),
              ('1-BUTANOL', 'WATER'),
              ('Toluene', 'WATER')]
all_aspen_models = ['UNIFAC', 'COSMOSAC']
all_ml_models = ['protocol_III']
```

**Output**: 3×3 grid with one row per system.

#### Section 3: Ternary Surface Analysis

**Purpose**: Visualize Gibbs energy surfaces for ternary systems.

```python
ternary_systems = [('CHLOROFORM', 'ACETONE', 'METHANOL')]
ternary_aspen_models = ['UNIFAC', 'COSMO-SAC']
ternary_ml_models = ['protocol_IV']
```

#### Customization

- **Marker intervals**: `MARKER_INTERVAL = 10`
- **Font size**: `fontsize = 14`
- **Colors**: Edit `aspen_styles` and `ml_styles` tuples
- **Resolution**: Adjust `dpi=300` in `savefig()`

> **Important**: Naming conventions differ between sources:
> - **ASPEN** (`outputs/aspen/`): `ETHANOL`, `N-BUTANOL`, `TOLUENE`
> - **ML** (`outputs/inference/`): `ETHANOL`, `1-BUTANOL`, `Toluene`
>
> Verify `aspen_sys_files` and `ml_sys_files` match their respective conventions to ensure proper comparison.
