import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)

import json
import random
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import deepthermomix.data.data_pipeline as dp
import deepthermomix.data.datasplit_scheme as dsm
import deepthermomix.train.trainer as tm
import deepthermomix.model.architecture as gm

def set_seed(seed):
    """Ensures reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="DTMPNN Training Script")
    
    # --- Data Paths ---
    parser.add_argument('--components_csv', type=str, default='development/datasets/components.csv')
    parser.add_argument('--raw_csv', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='outputs/training')
    
    # --- Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=21)
    
    # --- Architecture & Logic ---
    parser.add_argument('--constraint_type', type=str, default='none', choices=['soft', 'hard', 'none'])
    parser.add_argument('--gd_weight', type=float, default=0.01)
    parser.add_argument('--graph_hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=128)
    parser.add_argument('--graph_layers', type=int, default=3)
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading data from {args.raw_csv}...")
    
    pipeline = dp.DataPipeline(components_csv=args.components_csv)
    _, graph_list = pipeline.run_pipeline(raw_csv=args.raw_csv)

    random.shuffle(graph_list)
    train_set, val_set, test_set = dsm.system_disjoint_split(
        graph_list, 
        random_state=args.seed, 
        stratify_by_components=True
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['component_batch'])
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, follow_batch=['component_batch'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, follow_batch=['component_batch'])

    print(f"Data loaded: Train({len(train_set)}), Val({len(val_set)}), Test({len(test_set)})")
    
    # Logic: GD Backprop is ONLY needed for 'soft' constraints.
    # 'hard': constraints are IMPOSED permanently in the architecture
    # 'soft': constraints are enforced via loss function
    # 'none': no constraints are applied at all.
    if args.constraint_type == 'soft':
        include_gd = True
    else:
        include_gd = False

    config = vars(args)
    config['include_gd'] = include_gd
    config['device'] = device

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_root, 
        f"{args.constraint_type}_constraint",
        f"run_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'hyperparams.json'), 'w') as f:
        json.dump(config, f, indent=4)

    sample_batch = next(iter(train_loader))
    node_dim = sample_batch.x.shape[1]
    edge_dim = sample_batch.edge_attr.shape[1]

    model = gm.DTMPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        latent_dim=args.latent_dim,
        context_dim=args.context_dim,
        graph_layers=args.graph_layers,
        constraint_type=args.constraint_type
    ).to(device)

    trainer = tm.DTMPNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        include_gd=include_gd,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gd_weight=args.gd_weight,
        constraint_type=args.constraint_type
    )

    print(f"Starting training... Output: {output_dir}")
    
    trainer.train(
        epochs=args.epochs,
        save_dir=output_dir,
        log_file_path=os.path.join(output_dir, 'log.txt'),
        save_best=True,
        save_every=None
    )

    trainer.plot_history(save_path=os.path.join(output_dir, 'history.png'))
    print("Training complete.")

if __name__ == "__main__":
    main()