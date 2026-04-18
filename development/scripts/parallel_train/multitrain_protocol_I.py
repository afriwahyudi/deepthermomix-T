import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

runs = [
    ['--lr'                 , '0.001',
     '--weight_decay'       , '0', 
     '--constraint_type'    , 'hard', 
     '--gd_weight'          , '0', 
     '--graph_hidden_dim'   , '128', 
     '--latent_dim'         , '128', 
     '--context_dim'        , '128',
     '--graph_layers'       , '2',
     '--epochs'             , '200',
     '--raw_csv'            , 'development/datasets/binary_only/binary_no_aci.csv',
     '--batch_size'         , '1000',
     '--seed'               , '10021',],

    ['--lr'                 , '0.001', 
     '--weight_decay'       , '0', 
     '--constraint_type'    , 'soft', 
     '--gd_weight'          , '0.1', 
     '--graph_hidden_dim'   , '128', 
     '--latent_dim'         , '128', 
     '--context_dim'        , '128',
     '--graph_layers'       , '2',
     '--epochs'             , '200',
     '--raw_csv'            , 'development/datasets/binary_only/binary_no_aci.csv',
     '--batch_size'         , '1000',
     '--seed'               , '10021',],

    ['--lr'                 , '0.001', 
     '--weight_decay'       , '0', 
     '--constraint_type'    , 'none', 
     '--gd_weight'          , '0', 
     '--graph_hidden_dim'   , '128', 
     '--latent_dim'         , '128', 
     '--context_dim'        , '128',
     '--graph_layers'       , '2',
     '--epochs'             , '200',
     '--raw_csv'            , 'development/datasets/binary_only/binary_no_aci.csv',
     '--batch_size'         , '1000',
     '--seed'               , '10021',],
]

procs = [
    subprocess.Popen(['python', 'development/scripts/main_scripts/execute_training.py'] + args)
    for args in runs
]

for p in procs:
    p.wait()
