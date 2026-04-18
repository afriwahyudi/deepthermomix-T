import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
from rdkit import Chem
from torch_geometric.data import Data
from deepthermomix.inference.antoine_scrapper import AntoineEquation

class TernaryAnalyzer:
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

    def _prepare_batch(self, smiles_list, mole_fracs_batch):
        graphs = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            x, edge_index, edge_attr = self.pipeline._mol_to_graph(mol)
            graphs.append((x, edge_index, edge_attr))
            
        base_x_list = []
        base_edge_index_list = []
        base_edge_attr_list = []
        base_mol_batch_list = []
        node_offset = 0
        
        for i, (x, edge_index, edge_attr) in enumerate(graphs):
            base_x_list.append(x)
            base_edge_index_list.append(edge_index + node_offset)
            if edge_attr is not None:
                base_edge_attr_list.append(edge_attr)
            
            num_atoms = x.shape[0]
            base_mol_batch_list.append(torch.full((num_atoms,), i, dtype=torch.long))
            node_offset += num_atoms
            
        base_x = torch.cat(base_x_list, dim=0)
        base_edge_index = torch.cat(base_edge_index_list, dim=1)
        base_edge_attr = torch.cat(base_edge_attr_list, dim=0) if base_edge_attr_list else None
        base_mol_batch = torch.cat(base_mol_batch_list, dim=0)
        
        return base_x, base_edge_index, base_edge_attr, base_mol_batch

    def _generate_simplex_grid(self, steps=60):
        grid_points = []
        min_frac = 1e-9
        denominator = steps - 1

        for i in range(steps):
            for j in range(steps - i):
                x = i / denominator
                y = j / denominator
                z = 1.0 - x - y

                if x < min_frac: x = min_frac
                if y < min_frac: y = min_frac
                if z < min_frac: z = min_frac
                
                grid_points.append([x, y, z])
        
        arr = np.array(grid_points)
        arr_unique = np.unique(np.round(arr, decimals=10), axis=0)
        return arr_unique

    def predict_ternary_surface(self, smiles_list, steps=60):
        if len(smiles_list) != 3:
            raise ValueError("Must provide exactly 3 SMILES strings.")

        mole_fracs = self._generate_simplex_grid(steps)
        base_x, base_edge, base_attr, base_mol_batch = self._prepare_batch(smiles_list, mole_fracs)
        
        base_x = base_x.to(self.device)
        base_edge = base_edge.to(self.device)
        if base_attr is not None: base_attr = base_attr.to(self.device)
        base_mol_batch = base_mol_batch.to(self.device)

        chunk_size = 128
        g_excess_results = []
        g_mix_results = []
        ln_gamma_results = []
        
        for i in range(0, len(mole_fracs), chunk_size):
            chunk_fracs = mole_fracs[i : i + chunk_size]
            curr_batch_size = len(chunk_fracs)
            node_count = base_x.shape[0]
            batch_x_full = base_x.repeat(curr_batch_size, 1)
            edge_offsets = torch.arange(curr_batch_size, device=self.device) * node_count
            batch_edge_full = base_edge.repeat(1, curr_batch_size) + edge_offsets.repeat_interleave(base_edge.shape[1])
            batch_attr_full = base_attr.repeat(curr_batch_size, 1) if base_attr is not None else None
            mol_offsets = torch.arange(curr_batch_size, device=self.device) * 3
            batch_mol_batch_full = base_mol_batch.repeat(curr_batch_size) + mol_offsets.repeat_interleave(node_count)
            comp_batch_batch = torch.arange(curr_batch_size, device=self.device).repeat_interleave(3)
            comp_mole_frac_flat = torch.tensor(chunk_fracs, dtype=torch.float, device=self.device).view(-1)
            
            data = Data(
                x=batch_x_full,
                edge_index=batch_edge_full,
                edge_attr=batch_attr_full,
                mol_batch=batch_mol_batch_full,
                component_batch_batch=comp_batch_batch,
                component_mole_frac=comp_mole_frac_flat
                )
            
            ln_gamma_pred, _, _ = self.model(data)
            ln_gamma = ln_gamma_pred.view(curr_batch_size, 3)
            x_fracs_tensor = torch.tensor(chunk_fracs, dtype=torch.float, device=self.device)
            g_excess_batch = (x_fracs_tensor * ln_gamma).sum(dim=1)
            x_safe = x_fracs_tensor + 1e-16
            g_ideal_batch = (x_fracs_tensor * torch.log(x_safe)).sum(dim=1)
            g_mix_batch = g_excess_batch + g_ideal_batch

            ln_gamma_results.append(ln_gamma.cpu().detach().numpy())
            g_excess_results.append(g_excess_batch.cpu().detach().numpy())
            g_mix_results.append(g_mix_batch.cpu().detach().numpy())

        ln_gamma_total = np.concatenate(ln_gamma_results, axis=0)
        g_excess_total = np.concatenate(g_excess_results)
        g_mix_total = np.concatenate(g_mix_results)
        
        df = pd.DataFrame({
            'x1': mole_fracs[:, 0],
            'x2': mole_fracs[:, 1],
            'x3': mole_fracs[:, 2],
            'ln_gamma1': ln_gamma_total[:, 0],
            'ln_gamma2': ln_gamma_total[:, 1],
            'ln_gamma3': ln_gamma_total[:, 2],
            'g_mix_reduced': g_mix_total,
            'g_excess_reduced': g_excess_total
        })
        
        return mole_fracs, g_excess_total, g_mix_total, df

    def plot(self, smiles_list, save_path=None):
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 14
        })
        mole_fracs, g_excess, g_mix, df = self.predict_ternary_surface(smiles_list)
        names = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            can = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else s
            names.append(self.smiles_map.get(can, s))
        
        t = mole_fracs[:, 0] 
        l = mole_fracs[:, 1] 
        r = mole_fracs[:, 2] 
        
        fig = plt.figure(figsize=(18, 8))
        
        ax1 = fig.add_subplot(121, projection='ternary')
        self._setup_ternary_axis(ax1, t, l, r, g_excess, names, 'Excess Gibbs Energy ( $g^E / RT$ )')
        
        ax2 = fig.add_subplot(122, projection='ternary')
        self._setup_ternary_axis(ax2, t, l, r, g_mix, names, 'Gibbs Energy of Mixing ( $\Delta_{mix}g / RT$ )')

        plt.suptitle(f'{names[0]} / {names[1]} / {names[2]}', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ternary plot saved to {save_path}")
        
        return df
            
    def _setup_ternary_axis(self, ax, t, l, r, data, names, cbar_label):
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
        cntr = ax.tricontourf(t, l, r, data, levels=30, cmap='RdBu_r')
        ax.tricontour(t, l, r, data, levels=30, colors='k', linewidths=0.3, alpha=0.5)
        off = 0.10  
        ax.text(-off, 0.5 + off/2, 0.5 + off/2, names[0], 
                fontsize=12, ha='center', va='top', rotation=0)
        ax.text(0.5 + off/2, 0.5 + off/2, -off, names[1], 
                fontsize=12, ha='center', va='bottom', rotation=60)
        ax.text(0.5 + off/2, -off, 0.5 + off/2, names[2], 
                fontsize=12, ha='center', va='bottom', rotation=-60)
        cbar = plt.colorbar(cntr, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)