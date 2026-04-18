import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from torch_geometric.data import Data
from deepthermomix.inference.antoine_scrapper import AntoineEquation

class MultiComponentAnalyzer:
    def __init__(self, model, pipeline, device='cpu'):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.antoine = AntoineEquation()
        self.smiles_map = {}
        
        if hasattr(pipeline, 'solvent_id_to_smiles'):
            for sid, smi in pipeline.solvent_id_to_smiles.items():
                name = pipeline.solvent_id_to_name.get(sid, None)
                if name and smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                        self.smiles_map[can_smi] = name

    def _get_name(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return smiles
        
        can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)

        if can_smi in self.smiles_map:
            return self.smiles_map[can_smi]
        
        nist_name = self.antoine.get_stored_name(smiles)
        if nist_name:
            return nist_name
            
        return smiles

    def _prepare_batch_N(self, smiles_list, mole_fracs_batch):
        num_comps = len(smiles_list)
        batch_size = len(mole_fracs_batch)
        
        comp_graphs = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if not mol: raise ValueError(f"Invalid SMILES: {s}")
            x, edge_index, edge_attr = self.pipeline._mol_to_graph(mol)
            comp_graphs.append((x, edge_index, edge_attr))

        base_x_list = []
        base_edge_index_list = []
        base_edge_attr_list = []
        base_mol_batch_list = []
        node_offset = 0
        
        for i, (x, ei, ea) in enumerate(comp_graphs):
            base_x_list.append(x)
            base_edge_index_list.append(ei + node_offset)
            if ea is not None: base_edge_attr_list.append(ea)
            
            num_atoms = x.shape[0]
            base_mol_batch_list.append(torch.full((num_atoms,), i, dtype=torch.long))
            node_offset += num_atoms

        base_x = torch.cat(base_x_list, dim=0).to(self.device)
        base_edge_index = torch.cat(base_edge_index_list, dim=1).to(self.device)
        base_mol_batch = torch.cat(base_mol_batch_list, dim=0).to(self.device)
        
        if base_edge_attr_list:
            base_edge_attr = torch.cat(base_edge_attr_list, dim=0).to(self.device)
        else:
            base_edge_attr = None

        base_num_nodes = base_x.shape[0]

        x_final = base_x.repeat(batch_size, 1)
        
        edge_len = base_edge_index.shape[1]
        edge_shift = torch.arange(batch_size, device=self.device).repeat_interleave(edge_len) * base_num_nodes
        edge_index_repeated = base_edge_index.repeat(1, batch_size)
        edge_index_final = edge_index_repeated + edge_shift

        edge_attr_final = base_edge_attr.repeat(batch_size, 1) if base_edge_attr is not None else None

        mol_batch_len = base_mol_batch.shape[0]
        mol_batch_shift = torch.arange(batch_size, device=self.device).repeat_interleave(mol_batch_len) * num_comps
        mol_batch_final = base_mol_batch.repeat(batch_size) + mol_batch_shift

        component_batch_batch = torch.arange(batch_size, device=self.device).repeat_interleave(num_comps)
        component_mole_frac = torch.tensor(mole_fracs_batch, dtype=torch.float, device=self.device).view(-1).requires_grad_(True)

        data = Data(
            x=x_final,
            edge_index=edge_index_final,
            edge_attr=edge_attr_final,
            mol_batch=mol_batch_final,
            component_batch_batch=component_batch_batch,
            component_mole_frac=component_mole_frac
        )
        return data

    def scan_composition(self, smiles_list, target_idx=0, steps=50):
        num_comps = len(smiles_list)
        x_range = np.linspace(0, 1, steps)
        mole_fracs_list = []

        for x_target in x_range:
            fracs = np.zeros(num_comps)
            fracs[target_idx] = x_target
            
            remainder = 1.0 - x_target
            others_count = num_comps - 1
            if others_count > 0:
                share = remainder / others_count
                for i in range(num_comps):
                    if i != target_idx:
                        fracs[i] = share
            
            if np.sum(fracs) > 0:
                fracs = fracs / np.sum(fracs)
            mole_fracs_list.append(fracs)

        mole_fracs_batch = np.array(mole_fracs_list)

        data = self._prepare_batch_N(smiles_list, mole_fracs_batch)
        
        ln_gamma_flat, _, _ = self.model(data)
        
        ln_gamma = ln_gamma_flat.detach().view(steps, num_comps).cpu().numpy()
        gamma = np.exp(ln_gamma)

        return x_range, gamma, ln_gamma

    def plot_sweep(self, smiles_list, target_idx=0, steps=50, save_path=None, custom_names=None):
        x_axis, gamma, ln_gamma = self.scan_composition(smiles_list, target_idx, steps)
        
        names = []
        if custom_names and len(custom_names) == len(smiles_list):
            names = custom_names
        else:
            for s in smiles_list:
                names.append(self._get_name(s))
        
        target_name = names[target_idx]

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        for i in range(len(smiles_list)):
            ax[0].plot(x_axis, gamma[:, i], linewidth=2, label=f'$\gamma$ {names[i]}')
        
        ax[0].set_xlabel(f'mol frac, {target_name}', fontsize=12)
        ax[0].set_ylabel('Activity Coefficient ($\gamma$)', fontsize=12)
        ax[0].set_title(f'Activity Coefficients\n(Varying {target_name}, others equimolar)', fontsize=14)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlim(0, 1)

        for i in range(len(smiles_list)):
            ax[1].plot(x_axis, ln_gamma[:, i], linewidth=2, label=f'$ln \gamma$ {names[i]}')
            
        ax[1].set_xlabel(f'mol frac, {target_name}', fontsize=12)
        ax[1].set_ylabel('ln Activity Coefficient ($\ln \gamma$)', fontsize=12)
        ax[1].set_title(f'Log Activity Coefficients\n(Varying {target_name}, others equimolar)', fontsize=14)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        ax[1].set_xlim(0, 1)
        ax[1].axhline(0, color='k', linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")