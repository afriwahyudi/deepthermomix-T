import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from torch_geometric.data import Data
import torch.nn.functional as F
from deepthermomix.inference.antoine_scrapper import AntoineEquation

class VLEAnalyzer:
    def __init__(self, model, pipeline, device='cpu'):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.antoine = AntoineEquation()
        self.model.to(device)
        self.model.eval()
        self.smiles_map = {}
        if hasattr(pipeline, 'solvent_id_to_smiles'):
            for sid, smi in pipeline.solvent_id_to_smiles.items():
                name = pipeline.solvent_id_to_name.get(sid, None)
                if name and smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                        self.smiles_map[can_smi] = name

    def _prepare_single_point(self, smiles_list, mole_fracs, temperature):
        node_features_list = []
        edge_index_list = []
        edge_attr_list = []
        mol_batch_list = []
        node_offset = 0
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
                
            x, edge_index, edge_attr = self.pipeline._mol_to_graph(mol)
            edge_index = edge_index + node_offset
            
            node_features_list.append(x)
            edge_index_list.append(edge_index)
            if edge_attr is not None:
                edge_attr_list.append(edge_attr)
            
            num_atoms = x.shape[0]
            mol_batch_list.append(torch.full((num_atoms,), i, dtype=torch.long))
            node_offset += num_atoms

        x = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
        mol_batch = torch.cat(mol_batch_list, dim=0)
        num_comps = len(smiles_list)
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, mol_batch=mol_batch,
            component_batch_batch=torch.zeros(num_comps, dtype=torch.long),
            component_mole_frac=torch.tensor(mole_fracs, dtype=torch.float),
            temperature=torch.tensor([temperature], dtype=torch.float)
        )
        return data.to(self.device)

    def phase_calculation(self, smiles1, smiles2, T_kelvin, steps=51):
        x1_range = np.linspace(0, 1, steps)
        results = []
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        can1 = Chem.MolToSmiles(mol1, isomericSmiles=True) if mol1 else smiles1
        can2 = Chem.MolToSmiles(mol2, isomericSmiles=True) if mol2 else smiles2

        pipeline_name1 = self.smiles_map.get(can1, None)
        pipeline_name2 = self.smiles_map.get(can2, None)
        name1 = pipeline_name1 or self.antoine.get_stored_name(smiles1) or smiles1
        name2 = pipeline_name2 or self.antoine.get_stored_name(smiles2) or smiles2

        Psat1 = self.antoine.get_Psat(smiles1, T_kelvin, name=pipeline_name1)
        Psat2 = self.antoine.get_Psat(smiles2, T_kelvin, name=pipeline_name2)

        print(f"Generating VLE for {name1} / {name2} at {T_kelvin}K")
        
        for x1 in x1_range:
            x2 = 1.0 - x1
            
            data = self._prepare_single_point([smiles1, smiles2], [x1, x2], T_kelvin)
            ln_gamma_pred, _, _ = self.model(data)
            
            ln_gamma1 = ln_gamma_pred[0].item()
            ln_gamma2 = ln_gamma_pred[1].item()
            
            gamma1 = np.exp(ln_gamma1)
            gamma2 = np.exp(ln_gamma2)

            p1_partial = x1 * gamma1 * Psat1
            p2_partial = x2 * gamma2 * Psat2
            P_total = p1_partial + p2_partial
            
            y1 = p1_partial / P_total if P_total > 1e-9 else 0.0
            y2 = 1.0 - y1

            if x1 > 0.999: y1 = 1.0
            if x1 < 0.001: y1 = 0.0

            g_excess_reduced = (x1 * ln_gamma1 + x2 * ln_gamma2)
            term1 = x1 * np.log(x1) if x1 > 1e-9 else 0.0
            term2 = x2 * np.log(x2) if x2 > 1e-9 else 0.0
            g_mix_reduced = g_excess_reduced + (term1 + term2)

            results.append({
                'P': P_total, 
                'x1': x1                        , 'x2': x2,
                'y1': y1                        , 'y2': y2,
                'ln_gamma1': ln_gamma1          , 'ln_gamma2': ln_gamma2,
                'gamma1': gamma1                , 'gamma2': gamma2,
                'g_mix_reduced': g_mix_reduced,
                'g_excess_reduced': g_excess_reduced
            })

        df_final = pd.DataFrame(results)
        df_final.attrs['name1'] = name1
        df_final.attrs['name2'] = name2
        
        return df_final

    def plot_vle(self, df, title_prefix=None):
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix" 
        })
        fontsize = 14
        name1 = df.attrs.get('name1', 'Component 1')
        name2 = df.attrs.get('name2', 'Component 2')
        
        if title_prefix is None:
            title_prefix = f"{name1} / {name2}"

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

        # Plot 1: P-x-y
        ax[0].plot(df['x1'], df['P'], 'b', marker='o', markersize=4, label='x (liquid)', linewidth=2.5)
        ax[0].plot(df['y1'], df['P'], 'r', marker='o', markersize=4, label='y (vapor)', linewidth=2.5)
        ax[0].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
        ax[0].set_ylabel('Pressure (bar)', fontsize=fontsize)
        ax[0].set_title(f'{title_prefix}: P-x-y Diagram', fontsize=fontsize)
        ax[0].set_xlim(0, 1)
        ax[0].legend(fontsize=fontsize)
        ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

        # Plot 2: Activity Coefficients
        ax[1].plot(df['x1'], df['ln_gamma1'], 'green', marker='o', markersize=4, label=f'ln $\gamma$ {name1}', linewidth=2.5)
        ax[1].plot(df['x1'], df['ln_gamma2'], 'orange', marker='o', markersize=4, label=f'ln $\gamma$ {name2}', linewidth=2.5)
        ax[1].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
        ax[1].set_ylabel('ln $\gamma$', fontsize=fontsize)
        ax[1].set_title(f'{title_prefix}: Activity Coefficients', fontsize=fontsize)
        ax[1].set_xlim(0, 1)
        ax[1].legend(fontsize=fontsize)
        ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[1].axhline(0, color='red', linewidth=1.0, linestyle='--')

        # Plot 3: Gibbs Energy
        ax[2].plot(df['x1'], df['g_mix_reduced'], 'purple', label='$g_{mix} / RT$ (Total)', linewidth=2.5)
        ax[2].plot(df['x1'], df['g_excess_reduced'], 'k--', label='$g^E / RT$ (Excess)', linewidth=2.0, alpha=0.7)
        ax[2].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
        ax[2].set_ylabel('Energy ($RT$)', fontsize=fontsize)
        ax[2].set_title(f'{title_prefix}: Gibbs Energy', fontsize=fontsize)
        ax[2].set_xlim(0, 1)
        ax[2].legend(loc='best', fontsize=fontsize) 
        ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[2].axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()