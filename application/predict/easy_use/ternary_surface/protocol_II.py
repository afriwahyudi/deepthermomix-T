import subprocess
import sys
import os

# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 1. DEFINE YOUR SYSTEMS HERE
# ==========================================
# Format: (smiles1, smiles2, smiles3, temp)
SYSTEMS = [('CCO', 'O', 'c1ccccc1', '298.15'),
           ('CCO', 'c1ccccc1', 'C1CCCCC1', '298.15'),
           ('ClC(Cl)Cl', 'CC(C)=O', 'CO', '298.15'),
           ('ClC(Cl)Cl', 'CC(C)=O', 'O', '298.15'),]
# ==========================================
# 2. CONFIGURATION
# ==========================================
PROTOCOL = 'protocol_II'
BASE_MODEL_DIR = f'model_weights/{PROTOCOL}'
BASE_OUTPUT_DIR = f'outputs/inference/ternary/{PROTOCOL}'

# Format: ('constraint_arg', 'model_folder_name')
MODEL_CONFIGS = [
    ('hard', 'hard_constraint', 'hard'),
]
# ==========================================
# 3. GENERATION & EXECUTION
# ==========================================
runs = []
for constraint_type, model_sub, out_sub in MODEL_CONFIGS:
    for smiles1, smiles2, smiles3, temp in SYSTEMS:
        cmd_args = [
            '--smiles1',        smiles1,
            '--smiles2',        smiles2,
            '--smiles3',        smiles3,
            '--temp',           temp,
            '--model_dir',      os.path.join(BASE_MODEL_DIR, model_sub),
            '--constraint_type', constraint_type,
            '--output_dir', os.path.join(BASE_OUTPUT_DIR, out_sub),
        ]
        runs.append(cmd_args)

print(f"Queueing {len(runs)} jobs...")
procs = [
    subprocess.Popen(['python', 'application/predict/main_script/predict_ternary.py'] + args)
    for args in runs
]
for p in procs:
    p.wait()

print("All jobs finished.")