import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble
from deepthermomix.inference.ternary import TernaryAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ternary surface")
    
    parser.add_argument('--smiles1', type=str, required=True, 
                        help='SMILES Left')
    parser.add_argument('--smiles2', type=str, required=True, 
                        help='SMILES Right')
    parser.add_argument('--smiles3', type=str, required=True, 
                        help='SMILES Top')
    parser.add_argument('--temp', type=float, default=298.15, 
                        help='Temperature in Kelvin')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to the trained .pt model file')
    parser.add_argument('--constraint_type', type=str, default='hard',
                        help='Constraint type used during training')
    parser.add_argument('--components_csv', type=str, default='development/datasets/components.csv', 
                        help='Path to components database')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for the results')
    
    return parser.parse_args()


def main(): 
    args = parse_args() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    print("Loading pipeline and model...") 
    print(f"Loading model from: {args.model_dir}") 
    print("Initializing Data Pipeline...") 

    pipeline = DataPipeline(args.components_csv) 
    ensemble_model = load_ensemble(args.model_dir, constraint_type=args.constraint_type, device=device) 
    analyzer = TernaryAnalyzer(ensemble_model, pipeline, device) 
     
    smiles_list = [args.smiles1, args.smiles2, args.smiles3] 
    print(f"Generating ternary surface for {smiles_list}") 
    component_names = [analyzer.smiles_map.get(s, s) for s in smiles_list] 
    base_filename = '_'.join(component_names)
    
    png_path = os.path.join(args.output_dir, base_filename + '.png') 
    csv_path = os.path.join(args.output_dir, base_filename + '.csv')
    
    os.makedirs(os.path.dirname(png_path), exist_ok=True) 
     
    df = analyzer.plot(smiles_list, save_path=png_path) 
    
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


if __name__ == "__main__":
    main()