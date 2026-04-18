import subprocess
import sys
import os

# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 1. DEFINE YOUR SYSTEMS HERE
# ==========================================
# Format: (smiles_list, target_idx, steps)
SYSTEMS = [
    (['CCO', 'O', 'c1ccccc1'], '0', '50'),
    (['CCO', 'O', 'c1ccccc1'], '1', '50'),
    (['CCO', 'O', 'c1ccccc1'], '2', '50'),
]

# ==========================================
# 2. CONFIGURATION
# ==========================================
PROTOCOL = 'protocol_I'
BASE_MODEL_DIR = f'model_weights/{PROTOCOL}'
BASE_OUTPUT_DIR = f'outputs/inference/n_activity/{PROTOCOL}'

# Format: ('constraint_arg', 'model_folder_name', 'output_folder_name')
MODEL_CONFIGS = [
    ('hard', 'hard_constraint', 'hard'),
    ('soft', 'soft_constraint', 'soft'),
    ('none', 'none_constraint', 'none'),
]

# ==========================================
# 3. GENERATION & EXECUTION
# ==========================================
runs = []
for constraint_type, model_sub, out_sub in MODEL_CONFIGS:
    for smiles, target_idx, steps in SYSTEMS:
        cmd_args = [
            '--smiles',         *smiles,
            '--target_idx',     target_idx,
            '--steps',          steps,
            '--model_dir',      os.path.join(BASE_MODEL_DIR, model_sub),
            '--constraint_type', constraint_type,
            '--output_dir',     os.path.join(BASE_OUTPUT_DIR, out_sub),
        ]
        runs.append(cmd_args)

print(f"Queueing {len(runs)} jobs {PROTOCOL}...")
procs = [
    subprocess.Popen(['python', 'application/predict/main_script/predict_n_activity.py'] + args)
    for args in runs
]
for p in procs:
    p.wait()
print(f"All {PROTOCOL} jobs finished.")