import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(parent_dir)
from deepthermomix.inference.binary import VLEAnalyzer
from deepthermomix.data.data_pipeline import DataPipeline
from deepthermomix.model.ensemble_wrapper import load_ensemble

def parse_args():
    parser = argparse.ArgumentParser(description="Generate VLE Isotherm for a binary system")
    
    parser.add_argument('--smiles1', type=str, required=True, 
                        help='SMILES string for component 1')
    parser.add_argument('--smiles2', type=str, required=True, 
                        help='SMILES string for component 2')
    parser.add_argument('--temp', type=float, default=298.15, 
                        help='Temperature in Kelvin')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the trained .pt model file')
    parser.add_argument('--constraint_type', type=str, default='hard',
                        help='Constraint type used during training')
    parser.add_argument('--components_csv', type=str, default='development/datasets/components.csv', 
                        help='Path to components database')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for the results')
    parser.add_argument('--steps', type=int, default=100, 
                        help='Number of concentration steps')

    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    print("Initializing Data Pipeline...")

    pipeline = DataPipeline(args.components_csv)
    ensemble_model = load_ensemble(args.model_dir, constraint_type=args.constraint_type, device=device)
    analyzer = VLEAnalyzer(ensemble_model, pipeline)

    print(f"Generating Isotherm for:")
    print(f"  Component 1: {args.smiles1}")
    print(f"  Component 2: {args.smiles2}")
    print(f"  Temperature: {args.temp} K")

    try:
        df_vle = analyzer.phase_calculation(
            args.smiles1,
            args.smiles2,
            T_kelvin=args.temp,
            steps=args.steps
        )
    except Exception as e:
        print(f"Error generating isotherm: {e}")
        import inspect
        print("Expected arguments:", inspect.signature(analyzer.phase_calculation))
        return

    pipeline_name1 = analyzer.smiles_map.get(args.smiles1, None)
    pipeline_name2 = analyzer.smiles_map.get(args.smiles2, None)
    s1_clean = pipeline_name1 or analyzer.antoine.get_stored_name(args.smiles1) or args.smiles1
    s2_clean = pipeline_name2 or analyzer.antoine.get_stored_name(args.smiles2) or args.smiles2
    
    run_name = f"{s1_clean}_{s2_clean}"
    save_path = os.path.join(args.output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(save_path, 'vle_data.csv')
    df_vle.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

    try:
        analyzer.plot_vle(df_vle)
        plot_path = os.path.join(save_path, 'isotherm_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save plot. Error: {e}")

if __name__ == "__main__":
    main()