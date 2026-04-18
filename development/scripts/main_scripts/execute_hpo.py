import sys
import os
import argparse
import json
import random
import joblib
import torch
import numpy as np
import optuna
from pathlib import Path
from optuna.pruners import MedianPruner
from torch_geometric.loader import DataLoader
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)

import deepthermomix.data.data_pipeline as dp
import deepthermomix.data.datasplit_scheme as dsm
import deepthermomix.train.trainer as tm
import deepthermomix.model.architecture as gm

def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter search with Optuna")

    # --- Data Paths ---
    parser.add_argument('--components_csv', type=str, 
                        default='development/datasets/components.csv')
    parser.add_argument('--raw_csv', type=str, 
                        default='development/datasets/mixed/all_data.csv')
    parser.add_argument('--output_root', type=str, 
                        default='outputs/hpo')
    parser.add_argument('--db_path', type=str, 
                        default='outputs/hpo/master_hpo_study.db', help='Path to SQLite DB for Optuna')

    # --- HPO Config ---
    parser.add_argument('--n_trials', type=int, 
                        default=10)
    parser.add_argument('--epochs', type=int, 
                        default=5)
    parser.add_argument('--subset_size', type=int, 
                        default=None, help='Limit dataset size for faster HPO (e.g., 100000)')
    parser.add_argument('--seed', type=int, 
                        default=21)
    parser.add_argument('--batch_size', type=int, 
                        default=1024)

    # --- Logic ---
    parser.add_argument('--constraint_type', type=str, default='none', choices=['soft', 'hard', 'none'])

    return parser.parse_args()

class HPOObjective:
    def __init__(self, train_loader, val_loader, test_loader, device, constraint_type, include_gd, epochs, output_dir):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.constraint_type = constraint_type
        self.include_gd = include_gd
        self.epochs = epochs
        self.output_dir = output_dir

    def __call__(self, trial):
        log_dir = os.path.join(self.output_dir, "optuna_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'trial_{trial.number}.log')

        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
        if self.constraint_type in ['hard', 'none']:
            gd_weight = 1.0
        elif self.constraint_type == 'soft':
            gd_weight = trial.suggest_float('gd_weight', 1e-3, 1, log=True)
        
        graph_hidden_dim = trial.suggest_categorical('graph_hidden_dim', [16, 32, 64, 128, 256])
        latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64, 128, 256])
        context_dim = trial.suggest_categorical('context_dim', [16, 32, 64, 128, 256])
        graph_layers = trial.suggest_int('graph_layers', 2, 5)

        try:
            sample_batch = next(iter(self.train_loader))
            node_dim = sample_batch.x.shape[1]
            edge_dim = sample_batch.edge_attr.shape[1]

            model = gm.DTMPNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                graph_hidden_dim=graph_hidden_dim,
                latent_dim=latent_dim,
                context_dim=context_dim,
                graph_layers=graph_layers,
                constraint_type=self.constraint_type
            ).to(self.device)

            trainer = tm.DTMPNNTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                include_gd=self.include_gd,
                device=self.device,
                lr=lr,
                weight_decay=weight_decay,
                gd_weight=gd_weight
            )

            trainer.train(
                epochs=self.epochs,
                save_dir=None,
                log_file_path=log_file_path,
                save_best=False,
                save_every=None,
                optuna_trial=trial
            )
            
            best_val_loss = trainer.best_val_loss
            
            del model
            del trainer
            torch.cuda.empty_cache()

            return best_val_loss

        except optuna.TrialPruned:
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}", file=sys.stderr)
            torch.cuda.empty_cache()
            return float('inf')

def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs: {args.epochs}")

    include_gd = True if args.constraint_type == 'soft' else False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        args.output_root, 
        f"{args.constraint_type}_constraint", 
        f"hpo_run_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print("Loading Data Pipeline...")
    pipeline = dp.DataPipeline(components_csv=args.components_csv)
    _, graph_list = pipeline.run_pipeline(raw_csv=args.raw_csv)

    if args.subset_size and args.subset_size < len(graph_list):
        print(f"Subsetting data to first {args.subset_size} samples...")
        graph_list = graph_list[:args.subset_size]

    random.shuffle(graph_list)
    train, val, test = dsm.system_disjoint_split(
        graph_list, 
        random_state=args.seed, 
        stratify_by_components=True
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, follow_batch=['component_batch'])
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, follow_batch=['component_batch'])
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, follow_batch=['component_batch'])

    db_folder = os.path.dirname(args.db_path)
    if db_folder:
        os.makedirs(db_folder, exist_ok=True)
        
    study_name = f"DTMPNN_{args.constraint_type}_gd_{include_gd}"
    storage_url = f"sqlite:///{args.db_path}"

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=1)
    
    print(f"--- Starting Study: {study_name} ---")
    print(f"--- Storage: {storage_url} ---")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )

    objective = HPOObjective(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        constraint_type=args.constraint_type,
        include_gd=include_gd,
        epochs=args.epochs,
        output_dir=run_dir
    )

    try:
        study.optimize(
            objective, 
            n_trials=args.n_trials, 
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("--- HPO Interrupted. Saving current state... ---")

    print("\n--- Study Complete ---")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"Total Trials: {len(study.trials)}")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Pruned:    {len(pruned_trials)}")

    if len(completed_trials) > 0:
        best = study.best_trial
        print("\nBest Trial:")
        print(f"  Value (Val Loss): {best.value:.6f}")
        print("  Params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
            
        best_params_path = os.path.join(run_dir, 'best_hyperparams.json')
        with open(best_params_path, 'w') as f:
            json.dump(best.params, f, indent=4)
        print(f"Best params saved to: {best_params_path}")
    else:
        print("No trials completed successfully.")

if __name__ == "__main__":
    main()