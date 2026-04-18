import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch_scatter import scatter_add, scatter_mean

class MixMSELoss(nn.Module):
    """
    Mixture-invariant loss.

    Definition and rationale are documented in docs/model.md
    (see losses.py --> MixMSELoss).
    """
    def __init__(self):
        super(MixMSELoss, self).__init__()

    def forward(self, y_pred, batched_data):
        y_true = batched_data.component_ln_gammas
        batch_index = batched_data.component_batch_batch
        squared_errors = torch.pow(y_pred - y_true, 2)
        summed_mixture_errors = scatter_add(
            squared_errors, 
            batch_index, 
            dim=0
        )
        loss = torch.mean(summed_mixture_errors)
        
        return loss
    
class GibbsDuhemLoss(nn.Module):
    """
    Gibbs-Duhem consistency loss.

    Definition and rationale are documented in docs/model.md
    (see losses.py --> GibbsDuhemLoss).
    """
    def __init__(self, loss_type='optimized'):
        super(GibbsDuhemLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, data, prediction):
        mole_frac = data.component_mole_frac
        component_batch = data.component_batch_batch
        g_excess_i_RT = prediction

        if self.loss_type == 'explicit':
            gd_loss_batch = []

            for batch_idx in torch.unique(component_batch):
                mask = (component_batch == batch_idx)
                indices = torch.where(mask)[0]
                g_partial_local = g_excess_i_RT[indices]
                num_components = len(indices)
                jacobian_rows = []
                for i in range(num_components):
                    full_grad = autograd.grad(
                        outputs=g_partial_local[i],
                        inputs=mole_frac,
                        retain_graph=True,
                        create_graph=True
                    )[0]                   
                    local_gradients = full_grad[indices]
                    jacobian_rows.append(local_gradients)
                jacobian = torch.stack(jacobian_rows)
                x_i = mole_frac[indices]
                consistency_residual = torch.matmul(x_i.unsqueeze(0), jacobian).squeeze()
                residual_mean = torch.mean(consistency_residual)
                gd_loss_sample = torch.sum((consistency_residual - residual_mean) ** 2)
                gd_loss_batch.append(gd_loss_sample)

            gd_loss = torch.mean(torch.stack(gd_loss_batch))
        
        elif self.loss_type == 'optimized':

            weighted_energy = mole_frac * g_excess_i_RT
            g_excess_perRT_total = scatter_add(weighted_energy, component_batch, dim=0)
            (g_excess_total_perRT,) = autograd.grad(outputs=torch.sum(g_excess_perRT_total),
                                                           inputs=mole_frac,
                                                           retain_graph=True,create_graph=True)


            consistency_residual = g_excess_total_perRT - g_excess_i_RT
            residual_mean_per_mixture = scatter_mean(consistency_residual, component_batch, dim=0)
            residual_mean_expanded = residual_mean_per_mixture[component_batch]
            variance_penalty = (consistency_residual - residual_mean_expanded) ** 2
            gd_loss_batch = scatter_add(variance_penalty, component_batch, dim=0)
            
            gd_loss = torch.mean(gd_loss_batch)

        return gd_loss