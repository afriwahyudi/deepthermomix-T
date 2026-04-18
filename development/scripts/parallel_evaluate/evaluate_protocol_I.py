import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

constraints = ['hard', 'soft', 'none']
tests = [
    ('ternary_only/all_ternary.csv', 'ternary_test'),
    ('binary_only/aci_set1.csv', 'infinite_dilution_set1'),
    ('binary_only/aci_set2.csv', 'infinite_dilution_set2'),
]

runs = []
for dataset, output in tests:
    for constraint in constraints:
        runs.append([
            '--model_dir', f'model_weights/protocol_I/{constraint}_constraint',
            '--constraint_type', constraint,
            '--components_csv', 'development/datasets/components.csv',
            '--dataset_csv', f'development/datasets/{dataset}',
            '--output_dir', f'outputs/evaluation/protocol_I/{output}',
            '--batch_size', '100',
        ])

procs = [subprocess.Popen(['python', 'development/scripts/main_scripts/evaluate_performance.py'] + args) for args in runs]
for p in procs:
    p.wait()