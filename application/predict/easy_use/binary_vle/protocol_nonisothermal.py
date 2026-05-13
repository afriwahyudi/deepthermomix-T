import subprocess
import sys
import os
# --- Setup Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
# ==========================================
# 1. DEFINE YOUR SYSTEMS HERE
# ==========================================
SYSTEMS = [
    # Series 1 (alcohols / water)
    ('CO'       , 'O'),                             # METHANOL, WATER
    ('CCO'      , 'O'),                             # ETHANOL, WATER
    ('CCCO'     , 'O'),                             # 1-PROPANOL, WATER
    ('CCCCO'    , 'O'),                             # N-BUTANOL, WATER
    ('CCCCCO'   , 'O'),                             # 1-PENTANOL, WATER

    # Series 2 (aromatics / water)
    ('CC1=CC=CC=C1' , 'O'),                         # TOLUENE, WATER
    ('C1=CC=CC=C1'  , 'O'),                         # BENZENE, WATER
    ('C1CCCCC1'     , 'O'),                         # CYCLOHEXANE, WATER

    # Common solvent / water pairs
    ('CC(=O)C'      , 'O'),                         # ACETONE, WATER
    ('CC#N'         , 'O'),                         # ACETONITRILE, WATER
    ('CS(=O)C'      , 'O'),                         # DIMETHYL-SULFOXIDE, WATER
    ('CN1CCCC1=O'   , 'O'),                         # N-METHYL-2-PYRROLIDONE, WATER

    # Acids with water
    ('CC(=O)O'      , 'O'),                         # ACETIC-ACID, WATER
    ('CCC(=O)O'     , 'O'),                         # PROPIONIC-ACID, WATER
    ('O=CO'         , 'O'),                         # FORMIC-ACID, WATER

    # Alcohol + solvent (organic-organic)
    ('CO'       , 'CC(=O)C'),                       # METHANOL, ACETONE
    ('CCO'      , 'CC(=O)C'),                       # ETHANOL, ACETONE
    ('CCCO'     , 'CC(=O)C'),                       # 1-PROPANOL, ACETONE
    ('CCO'      , 'C1=CC=CC=C1'),                   # ETHANOL, BENZENE
    ('CCO'      , 'CC1=CC=CC=C1'),                  # ETHANOL, TOLUENE

    # Nonpolar organics
    ('C1=CC=CC=C1' , 'CC1=CC=CC=C1'),               # BENZENE, TOLUENE
    ('C1=CC=CC=C1' , 'C1CCCCC1'),                   # BENZENE, CYCLOHEXANE
    ('CC1=CC=CC=C1', 'C1CCCCC1'),                   # TOLUENE, CYCLOHEXANE

    # Polar organics / special pairs
    ('C1=CC=C(C=C1)O' , 'O'),                       # PHENOL, WATER
    ('C(C(CO)O)O'     , 'O'),                       # GLYCEROL, WATER
    ('CS(=O)C'        , 'CO'),                      # DIMETHYL-SULFOXIDE, METHANOL
    ('CC#N'           , 'CO'),                      # ACETONITRILE, METHANOL
    ('CC#N'           , 'CC(=O)C'),                 # ACETONITRILE, ACETONE

    # Heterocycles / aromatics
    ('C1=CC=NC=C1'    , 'O'),                       # PYRIDINE, WATER
    ('C1=CC=C(C=C1)N' , 'O'),                       # ANILINE, WATER
    ('C1=CSC=C1'      , 'CC1=CC=CC=C1'),            # THIOPHENE, TOLUENE
    ('C1=CC=C(C=C1)[N+](=O)[O-]' , 'CC1=CC=CC=C1'), # NITROBENZENE, TOLUENE

    # Halogen / sulfur with aromatics
    ('ClC(Cl)Cl'      , 'C1=CC=CC=C1'),             # CHLOROFORM, BENZENE
    ('C(=S)=S'        , 'C1=CC=CC=C1'),             # CARBON-DISULFIDE, BENZENE
]
# ==========================================
# 2. CONFIGURATION
# ==========================================
PROTOCOL = 'protocol_I'
TEMPS = ['298.15', '323.15', '348.15', '373.15', '398.15']
STEPS = '101'
BASE_MODEL_DIR = f'model_weights/{PROTOCOL}'
BASE_OUTPUT_DIR = f'outputs/inference/pxy/{PROTOCOL}'
# Format: ('constraint_arg', 'model_folder_name', 'output_folder_name')
MODEL_CONFIGS = [
    ('hard', 'hard_constraint', 'hard')
]
# ==========================================
# 3. GENERATION & EXECUTION
# ==========================================
runs = []
for temp in TEMPS:
    for constraint_type, model_sub, out_sub in MODEL_CONFIGS:
        for s1, s2 in SYSTEMS:
            cmd_args = [
                '--smiles1',         s1,
                '--smiles2',         s2,
                '--temp',            temp,
                '--steps',           STEPS,
                '--model_dir',       os.path.join(BASE_MODEL_DIR, model_sub),
                '--constraint_type', constraint_type,
                '--output_dir',      os.path.join(BASE_OUTPUT_DIR, temp, out_sub),
            ]
            runs.append(cmd_args)
print(f"Queueing {len(runs)} jobs {PROTOCOL}...")
procs = [
    subprocess.Popen(['python', 'application/predict/main_script/predict_binary.py'] + args)
    for args in runs
]
for p in procs:
    p.wait()
print(f"All {PROTOCOL} jobs finished.")