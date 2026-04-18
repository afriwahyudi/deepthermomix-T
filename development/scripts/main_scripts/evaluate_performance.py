import sys
import os
import argparse
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)

import deepthermomix.data.data_pipeline as dp
import deepthermomix.data.datasplit_scheme as dsm
from deepthermomix.inference.utils import ComputeMetric
from deepthermomix.model.ensemble_wrapper import load_ensemble

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation Script")
    
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='Directory containing the .pt model files')
    
    parser.add_argument('--constraint_type', type=str, default='soft', choices=['soft', 'hard', 'none'])
    parser.add_argument('--components_csv', type=str, default='development/datasets/components.csv')
    parser.add_argument('--dataset_csv', type=str, required=True, help='Path to dataset (e.g., all_data.csv)')
    
    parser.add_argument('--split_type', type=str, default='system', choices=['system', 'random', 'none'],
                        help='How to split data. "none" uses the full dataset for testing')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--filter_n_components', type=int, default=None, choices=[1, 2, 3])
    
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation_ensemble')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset_csv}")
    pipeline = dp.DataPipeline(components_csv=args.components_csv)
    _, graph_list = pipeline.run_pipeline(raw_csv=args.dataset_csv)
    
    if args.split_type == 'system':
        print(f"Applying System Disjoint Split (Seed: {args.seed})...")
        random.shuffle(graph_list)
        _, _, test_set = dsm.system_disjoint_split(graph_list, random_state=args.seed, stratify_by_components=True)
        print(f"Test set size: {len(test_set)}")
    elif args.split_type == 'random':
        random.shuffle(graph_list)
        split_idx = int(len(graph_list) * (1 - args.test_ratio))
        test_set = graph_list[split_idx:]
        print(f"Random split test set size: {len(test_set)}")
    else:
        test_set = graph_list
        print(f"Using full dataset. Size: {len(test_set)}")

    if args.filter_n_components:
        DILUTION_TOL = 1e-4 
        print(f"Filtering for EFFECTIVE {args.filter_n_components}-component systems (Tol: {DILUTION_TOL})...")
        
        filtered_test = []
        for data in test_set:
            if hasattr(data, 'component_mole_frac'):
                fractions = data.component_mole_frac
                nominal_n = fractions.shape[0]
                
                if torch.is_tensor(fractions):
                    max_x = fractions.max().item()
                else:
                    max_x = max(fractions)

                if max_x >= (1.0 - DILUTION_TOL):
                    effective_n = 1
                else:
                    effective_n = nominal_n
                
                if effective_n == args.filter_n_components:
                    filtered_test.append(data)
                    
        test_set = filtered_test
        print(f"Filtered test set size: {len(test_set)}")

    if len(test_set) == 0:
        print("Error: Test set is empty.")
        return

    ensemble_model = load_ensemble(args.model_dir, constraint_type=args.constraint_type, device=device)
    ensemble_model.eval()
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, follow_batch=['component_batch'])
    print("Initializing ComputeMetric with Ensemble...")
    evaluator = ComputeMetric(ensemble_model, test_loader, device=device)
    
    df_formatted, df_raw, rmse, mae, r2, gd_loss = evaluator.run_evaluation()
    boot = evaluator.bootstrap_metrics(df_raw)
    mean_uncertainty = df_raw['uncertainty'].mean()
    median_uncertainty = df_raw['uncertainty'].median()
    std_uncertainty = df_raw['uncertainty'].std()

    dir_name = os.path.basename(os.path.normpath(args.model_dir))
    data_name = os.path.splitext(os.path.basename(args.dataset_csv))[0]
    save_dir = os.path.join(args.output_dir, f"Ensemble_{dir_name}_on_{data_name}")
    if args.filter_n_components: save_dir += f"_{args.filter_n_components}comp"
    
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Ensemble Size: {len(ensemble_model.models)}\n")
        f.write(f"RMSE: {rmse:.6f} +/- {boot['rmse_std']:.6f}\n")
        f.write(f"MAE: {mae:.6f} +/- {boot['mae_std']:.6f}\n")
        f.write(f"R2: {r2:.6f} +/- {boot['r2_std']:.6f}\n")
        f.write(f"GD_Loss: {gd_loss:.6e}\n")
        f.write(f"Mean Uncertainty: {mean_uncertainty:.6f}\n")
        f.write(f"Median Uncertainty: {median_uncertainty:.6f}\n")
        f.write(f"Std Uncertainty: {std_uncertainty:.6f}\n")

    df_formatted.to_csv(os.path.join(save_dir, "predictions_formatted.csv"), index=False)
    df_raw.to_csv(os.path.join(save_dir, "predictions_raw.csv"), index=False)

    # Parity Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_raw, x='ln_gamma_exp', y='ln_gamma_pred', alpha=0.4, color='purple')

    min_val = min(df_raw['ln_gamma_exp'].min(), df_raw['ln_gamma_pred'].min()) - 0.5
    max_val = max(df_raw['ln_gamma_exp'].max(), df_raw['ln_gamma_pred'].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel("Experimental ln($\gamma$)", fontsize=16)
    plt.ylabel("Ensemble Predicted ln($\gamma$)", fontsize=16)
    plt.title(f"Ensemble Parity Plot\nRMSE: {rmse:.3f} | R2: {r2:.3f}", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=11)

    plt.savefig(os.path.join(save_dir, "parity_plot.png"), dpi=300)
    plt.close()
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == "__main__":
    main()