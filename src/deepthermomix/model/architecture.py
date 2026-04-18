import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add

class MPNNLayer(MessagePassing):
    """
    Custom Message Passing Layer.

    Definition are documented in docs/model.md
    (see architecture.py --> MPNNLayer & MPNNBlock).
    """
    def __init__(self, node_dim, edge_dim, graph_hidden_dim):
        super(MPNNLayer, self).__init__(aggr="add")
        self.lin_node = nn.Linear(node_dim, graph_hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, graph_hidden_dim)

        self.message_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim + graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
            )

        self.update_mlp = nn.Sequential(
            nn.Linear(graph_hidden_dim + graph_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_hidden_dim)
            )

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x_orig):
        update_input = torch.cat([aggr_out, x_orig], dim=-1)
        return self.update_mlp(update_input)
        
    def forward(self, x, edge_index, edge_attr, mol_batch):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=0)
        x_transformed = self.lin_node(x)
        edge_attr_transformed = self.lin_edge(edge_attr)
        node_emb = self.propagate(edge_index, x=x_transformed, 
                                  edge_attr=edge_attr_transformed, x_orig=x_transformed)

        return node_emb

class MPNNBlock(nn.Module):
    """
    Stackable MPNN Layer.

    Definition are documented in docs/model.md
    (see architecture.py --> MPNNLayer & MPNNBlock).
    """
    def __init__(self, node_dim, edge_dim, graph_hidden_dim, num_layers):
        super(MPNNBlock, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(MPNNLayer(node_dim, edge_dim, graph_hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(MPNNLayer(graph_hidden_dim, edge_dim, graph_hidden_dim))
        if num_layers > 1:
            self.skip_weights = nn.Parameter(torch.ones(num_layers - 1))

    def forward(self, x, edge_index, edge_attr, mol_batch):
        node_emb          = self.layers[0](x, edge_index, edge_attr, mol_batch)
        for i in range(1, self.num_layers):
            node_emb_new  = self.layers[i](node_emb, edge_index, edge_attr, mol_batch)
            node_emb      = node_emb_new + torch.relu(self.skip_weights[i-1]) * node_emb
        comp_emb = global_mean_pool(node_emb, mol_batch)
                
        return node_emb, comp_emb
    
class DeepThermoMix(nn.Module):
    """
    MLP-parameterized local composition hypothesis.

    Definition are documented in docs/model.md
    (see architecture.py --> DeepThermoMix).
    """
    def __init__(self, component_emb_dim, latent_dim, context_dim):
        super(DeepThermoMix, self).__init__()
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(component_emb_dim, context_dim),
            nn.SiLU(),          
            nn.Linear(context_dim, context_dim)
        )
        
        gate_input_dim = component_emb_dim + context_dim + 1 + 1
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, comp_emb, mole_frac, component_batch_batch, temperature): 
        mole_frac_view = mole_frac.view(-1, 1)
        temp_normalized = (temperature[component_batch_batch] / 273.15)  # [num_nodes, 1]
        weighted_comp_emb = comp_emb * mole_frac_view
        mixture_state = scatter_add(weighted_comp_emb, component_batch_batch, dim=0)
        mixture_context = self.interaction_mlp(mixture_state)
        mixture_context_expanded = mixture_context[component_batch_batch]

        gate_input = torch.cat([comp_emb, 
                                mole_frac_view, 
                                mixture_context_expanded, 
                                temp_normalized],
                                  dim=-1)
        
        mixture_latent_vectors = self.gate_mlp(gate_input)

        return mixture_latent_vectors

    def forward_ideal(self, comp_emb, mole_frac, component_batch_batch, temperature):
        mole_frac_view = mole_frac.view(-1, 1)
        temp_normalized = (temperature[component_batch_batch] / 273.15)  # [num_nodes, 1]
        pure_state = comp_emb
        pure_context = self.interaction_mlp(pure_state)
        gate_input = torch.cat([comp_emb,
                                mole_frac_view, 
                                pure_context,
                                temp_normalized],
                                  dim=-1)
        
        ideal_latent_vectors = self.gate_mlp(gate_input)
        return ideal_latent_vectors

class DTMPNN(nn.Module):
    """
    Deep Thermodynamics Mixing 
    Message Passing Neural Network

    Definition are documented in docs/model.md
    (see architecture.py --> DTMPNN).
    """
    def __init__(self, 
                 node_dim, 
                 edge_dim, 
                 graph_hidden_dim, 
                 latent_dim, 
                 context_dim,
                 graph_layers,
                 constraint_type='hard'):
        super(DTMPNN, self).__init__()
        self.constraint_type    = constraint_type
        self.graph_block        = MPNNBlock(node_dim, edge_dim, graph_hidden_dim, num_layers=graph_layers)
        self.mixture_layer      = DeepThermoMix(graph_hidden_dim, latent_dim, context_dim)
        self.output_layer_soft  = nn.Linear(latent_dim, 1)
        self.output_layer_hard  = nn.Linear(latent_dim, 1)

    def forward(self, data):
        # ====== IMPORTANT : DO NOT REMOVE THIS BLOCK ======
        temperature = data.temperature.view(-1, 1)
        mole_frac = data.component_mole_frac.clone().detach()
        mole_frac.requires_grad_(True)
        data.component_mole_frac = mole_frac
        # ==================================================

        node_emb, comp_emb = self.graph_block(data.x, data.edge_index, 
                                              data.edge_attr, data.mol_batch)
        mixture_latent_vectors = self.mixture_layer(comp_emb, 
                                                        mole_frac,
                                                        data.component_batch_batch,
                                                        temperature)
             
        if self.constraint_type in ['none', 'soft']:
            ln_gammas_calc = self.output_layer_soft(mixture_latent_vectors).squeeze(-1)
            prediction = ln_gammas_calc

        elif self.constraint_type == 'hard':
            ideal_latent_vectors    = self.mixture_layer.forward_ideal(comp_emb, mole_frac, data.component_batch_batch, temperature)
            pseudo_output_mixture   = self.output_layer_hard(mixture_latent_vectors).squeeze(-1)
            pseudo_output_ideal     = self.output_layer_hard(ideal_latent_vectors).squeeze(-1)
            excess_term             = mole_frac * (pseudo_output_mixture - pseudo_output_ideal)
            g_excess_total_perRT    = scatter_add(excess_term, data.component_batch_batch, dim=0)

            (dgE_perRT_dx,) = autograd.grad(outputs=torch.sum(g_excess_total_perRT),
                                      inputs=mole_frac,
                                      retain_graph=True,
                                      create_graph=True)
            
            term_A = g_excess_total_perRT[data.component_batch_batch]
            term_B = dgE_perRT_dx
            x_times_gradient = mole_frac * dgE_perRT_dx
            correction_term = scatter_add(x_times_gradient, data.component_batch_batch, dim=0)
            term_C = correction_term[data.component_batch_batch]
            prediction = term_A + term_B - term_C

        return prediction, mixture_latent_vectors, comp_emb 