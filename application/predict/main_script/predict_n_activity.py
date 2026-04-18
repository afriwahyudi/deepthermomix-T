import argparse
import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble
from deepthermomix.inference.n_component import MultiComponentAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Activity Coefficients for N-Component Mixtures")
    
    parser.add_argument('--smiles', type=str, nargs='+', required=True, 
                        help='List of SMILES strings (space separated)')
    parser.add_argument('--target_idx', type=int, default=0, 
                        help='Index of the component to vary (0-based)')
    parser.add_argument('--steps', type=int, default=50, 
                        help='Number of steps in the composition sweep')
    parser.add_argument('--model_dir', type=str,
                        help='Path to the trained .pt model file')
    parser.add_argument('--constraint_type', type=str, default='hard',
                        help='Constraint type used during training')
    parser.add_argument('--components_csv', type=str, default='development/datasets/components.csv', 
                        help='Path to components database')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for the results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    print("Initializing Data Pipeline...")
    
    pipeline = DataPipeline(args.components_csv)
    model = load_ensemble(args.model_dir, constraint_type=args.constraint_type, device=device)
    analyzer = MultiComponentAnalyzer(model, pipeline, device)
    
    if len(args.smiles) < 2:
        print("Warning: You provided fewer than 2 components. A sweep requires at least a binary mixture.")
    
    if args.target_idx >= len(args.smiles):
        raise ValueError(f"Target index {args.target_idx} is out of bounds for {len(args.smiles)} components.")
    
    target_comp = args.smiles[args.target_idx]
    print(f"Scanning composition for component {args.target_idx} ({target_comp}) in mixture: {args.smiles}")
    
    component_names = [analyzer.smiles_map.get(s, s) for s in args.smiles]
    swept_name = component_names.pop(args.target_idx)
    ordered_names = [swept_name] + component_names
    filename = '_'.join(ordered_names) + '.png'
    
    output_path = os.path.join(args.output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    analyzer.plot_sweep(
        args.smiles, 
        target_idx=args.target_idx, 
        steps=args.steps, 
        save_path=output_path
    )

if __name__ == "__main__":
    main()