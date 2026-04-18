from typing import List, Tuple, Dict, Union
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import torch
import copy
import pandas as pd
import numpy as np

def get_effective_n(
    n_components: int, 
    fractions: Union[List[float], np.ndarray, torch.Tensor], 
    tol: float
) -> int:
    """
    Helper to determine effective N. 
    If max(fractions) >= 1.0 - tol, treat as N=1 (infinite dilution).
    """
    if fractions is None:
        return n_components

    if hasattr(fractions, '__len__') and len(fractions) == 0:
        return n_components
    
    if isinstance(fractions, torch.Tensor):
        max_x = fractions.max().item()

    elif isinstance(fractions, (np.ndarray, list)):
        max_x = np.max(fractions)

    else:
        return n_components

    if max_x >= (1.0 - tol):
        return 1
    
    return n_components

def system_disjoint_split(
    data: Union[List[Data], pd.DataFrame],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by_components: bool = False,
    dilution_tol: float = 1e-4) -> Tuple:
    """
    Split PyG Data graphs OR a Pandas DataFrame into disjoint train/val/test sets by system_id.
    
    Args:
        data: List of PyG Data objects or Pandas DataFrame.
        test_size: Fraction for test set.
        val_size: Fraction for validation set.
        random_state: Seed.
        stratify_by_components: Maintain N-ary system proportions.
        dilution_tol: If max(x_i) >= 1.0 - tol, treat system as N=1 (Pure/Infinite Dilution).
    """
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        if 'system_id' not in data.columns:
            raise ValueError("DataFrame must contain a 'system_id' column.")
        
        if stratify_by_components:
            if 'component_names' in data.columns:
                comp_col = 'component_names'
            elif 'component_list' in data.columns:
                comp_col = 'component_list'
            else:
                raise ValueError("DataFrame must contain 'component_names' or 'component_list' for stratification.")
            
            frac_col = None

            for candidate in ['component_mole_frac', 'mole_fractions', 'x', 'composition']:
                if candidate in data.columns:
                    frac_col = candidate
                    break
            
            cols_to_fetch = [comp_col]
            if frac_col:
                cols_to_fetch.append(frac_col)
                
            system_info = data.groupby('system_id')[cols_to_fetch].first()
            system_ids = system_info.index.values
            
            counts = []
            for _, row in system_info.iterrows():
                n = len(row[comp_col])
                f = row[frac_col] if frac_col else None
                counts.append(get_effective_n(n, f, dilution_tol))
            
            component_counts = np.array(counts)
            
        else:
            system_ids = data['system_id'].unique()
            component_counts = None

    elif isinstance(data, list):
        system_map = {} 
        
        for g in data:
            if g.system_id not in system_map:
                if hasattr(g, 'component_names'):
                    nominal_n = len(g.component_names)
                elif hasattr(g, 'num_components'):
                    nominal_n = int(g.num_components)
                else:
                    nominal_n = 2 
                
                fractions = None
                if hasattr(g, 'component_mole_frac'):
                    fractions = g.component_mole_frac
                elif hasattr(g, 'mole_fractions'):
                    fractions = g.mole_fractions
                
                system_map[g.system_id] = (nominal_n, fractions)
        
        system_ids = np.array(list(system_map.keys()))
        
        if stratify_by_components:
            component_counts = []
            for sid in system_ids:
                nominal_n, frac = system_map[sid]
                eff_n = get_effective_n(nominal_n, frac, dilution_tol)
                component_counts.append(eff_n)
            component_counts = np.array(component_counts)
        else:
            component_counts = None
    else:
        raise TypeError("Input must be a list of PyG Data objects or a pandas DataFrame.")
    
    if stratify_by_components:
        train_val_ids, test_ids, train_val_c, _ = train_test_split(
            system_ids, component_counts,
            test_size=test_size, 
            random_state=random_state,
            stratify=component_counts
        )
        relative_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val_size, 
            random_state=random_state,
            stratify=train_val_c
        )
    else:
        train_val_ids, test_ids = train_test_split(
            system_ids, test_size=test_size, random_state=random_state
        )
        relative_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=relative_val_size, random_state=random_state
        )
    
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)
    
    if is_dataframe:
        system_id_col = data['system_id']
        train_data = data[system_id_col.isin(train_ids_set)].copy()
        val_data = data[system_id_col.isin(val_ids_set)].copy()
        test_data = data[system_id_col.isin(test_ids_set)].copy()
    
        if stratify_by_components:
            dist_cols = [comp_col]
            if frac_col: dist_cols.append(frac_col)
            
            def get_df_dist(df):
                grouped = df.groupby('system_id')[dist_cols].first()
                d_counts = []
                for _, r in grouped.iterrows():
                    n = len(r[comp_col])
                    f = r[frac_col] if frac_col else None
                    d_counts.append(get_effective_n(n, f, dilution_tol))
                
                d = {}
                for c in d_counts: d[c] = d.get(c, 0) + 1
                return dict(sorted(d.items()))

            print("Effective Component Distribution (N=1 includes x->1):")
            print(f"  Train: {get_df_dist(train_data)}")
            print(f"  Val:   {get_df_dist(val_data)}")
            print(f"  Test:  {get_df_dist(test_data)}")

    else:
        train_data = []
        val_data = []
        test_data = []
        
        for g in data:
            if g.system_id in train_ids_set:
                train_data.append(g)
            elif g.system_id in val_ids_set:
                val_data.append(g)
            else:
                test_data.append(g)
    
        if stratify_by_components:
            def get_list_dist(d_list):
                counts = []
                for g in d_list:
                    n = len(g.component_names) if hasattr(g, 'component_names') else 2
                    f = getattr(g, 'component_mole_frac', getattr(g, 'mole_fractions', None))
                    counts.append(get_effective_n(n, f, dilution_tol))
                
                d = {}
                for c in counts: d[c] = d.get(c, 0) + 1
                return dict(sorted(d.items()))
        
            print("Effective Component Distribution (N=1 includes x->1):")
            print(f"  Train: {get_list_dist(train_data)}")
            print(f"  Val:   {get_list_dist(val_data)}")
            print(f"  Test:  {get_list_dist(test_data)}")
    
    print(f"\nDatapoints -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Unique systems -> Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return train_data, val_data, test_data