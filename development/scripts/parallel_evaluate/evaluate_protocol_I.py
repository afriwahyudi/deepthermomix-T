import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

constraints = ['hard', 'soft', 'none']
tests = [
    ('non_isothermal/ternary/kdb_ternary.csv', 'kdb_ternary'),
    ('non_isothermal/ternary/cosmo_ternary.csv', 'cosmo_ternary'),
    ('non_isothermal/binary/aci_set2.csv', 'aci'),
]

runs = []
for dataset, output in tests:
    for constraint in constraints:
        runs.append([
            '--model_dir', f'model_weights/protocol_I/{constraint}_constraint',
            '--constraint_type', constraint,
            '--components_csv', 'development/datasets/component_set_unified.csv',
            '--dataset_csv', f'development/datasets/{dataset}',
            '--output_dir', f'outputs/evaluation/protocol_I/{output}',
            '--batch_size', '100',
        ])

procs = [subprocess.Popen(['python', 'development/scripts/main_scripts/evaluate_performance.py'] + args) for args in runs]
for p in procs:
    p.wait()