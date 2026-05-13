import win32com.client as win32
import os
import pandas as pd
import numpy as np
import psutil
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)

def kill_all_aspen_processes():
    aspen_process_names = ['AspenPlus.exe', 'Apwn.exe', 'AspenPlusGUI.exe', 'APwnMain.exe', 'Aspen.exe']
    killed_count = 0
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] in aspen_process_names:
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if killed_count > 0:
        time.sleep(2)

class AspenVLE:
    MODEL_MAP = {
        'COSMOSAC'  : 'BINRY-1',
        'NRTL'      : 'BINRY-2',
        'UNIFAC'    : 'BINRY-3',
        'UNIQUAC'   : 'BINRY-4',
        'WILSON'    : 'BINRY-5'
    }

    def get_VLE_from_aspen(self, 
                           chemical_list, 
                           npoint=50, 
                           temperature=298.15,
                           model_name='NRTL'):
        
        bkp_path = os.path.abspath(os.path.join(parent_dir, 'application/aspen/aspen_bkp_files/binary_analysis.bkp'))
        
        clean_name = model_name.upper()
        
        if clean_name in self.MODEL_MAP:
            analysis_id = self.MODEL_MAP[clean_name]
        elif clean_name in self.MODEL_MAP.values():
            analysis_id = clean_name
            clean_name = next(key for key, val in self.MODEL_MAP.items() if val == clean_name)
        else:
            print(f"Warning: Model '{model_name}' not found in map. Defaulting to BINRY-1.")
            analysis_id = 'BINRY-1'
            
        print(f"Selected Model: {clean_name} -> ID: {analysis_id}")

        aspen = win32.Dispatch('Apwn.Document')
        aspen.InitFromArchive2(bkp_path)
        aspen.Visible = 0  
        aspen.SuppressDialogs = 1
        
        print(f"Injecting components: {chemical_list}...")

        try:
            for idx, comp_name in enumerate(chemical_list, 1):
                solv_id = f'SOLV{idx}'
                node_path = f'/Data/Components/Specifications/Input/DBNAME1/{solv_id}'
                aspen.Tree.FindNode(node_path).Value = comp_name.upper()
            
            aspen.Tree.FindNode(f'/Data/Properties/Analysis/{analysis_id}/Input/CNPOINT').Value = npoint
            aspen.Tree.FindNode(f'/Data/Properties/Analysis/{analysis_id}/Input/TLIST/#0').Value = temperature
            
            aspen.Visible = 1 
            print("Running Simulation...")
            aspen.Engine.Run2()
            
            base_path = f'/Data/Properties/Analysis/{analysis_id}/Output/Prop Data/PROPTAB'
            liq_prefix = "LIQUID1"
            if aspen.Tree.FindNode(f'{base_path}/LIQUID1 MOLEFRAC SOLV1/1') is None:
                liq_prefix = "LIQUID"
                if aspen.Tree.FindNode(f'{base_path}/LIQUID MOLEFRAC SOLV1/1') is None:
                    raise ValueError(f"Aspen output table empty for {analysis_id}. Simulation failed.")

            x1, y1, solv1_gam, solv2_gam, equi_p = [], [], [], [], []
            
            for i in tqdm(range(npoint + 1), desc="Extracting data", unit="pt"):
                idx = i + 1
                x1.append(aspen.Tree.FindNode(f'{base_path}/{liq_prefix} MOLEFRAC SOLV1/{idx}').Value)
                y1.append(aspen.Tree.FindNode(f'{base_path}/VAPOR MOLEFRAC SOLV1/{idx}').Value)
                solv1_gam.append(aspen.Tree.FindNode(f'{base_path}/{liq_prefix} GAMMA SOLV1/{idx}').Value)
                solv2_gam.append(aspen.Tree.FindNode(f'{base_path}/{liq_prefix} GAMMA SOLV2/{idx}').Value)
                equi_p.append(aspen.Tree.FindNode(f'{base_path}/TOTAL PRES/{idx}').Value)
                
        except Exception as e:
            aspen.Close()
            raise e
            
        aspen.Close()
        
        x1_arr = np.array(x1, dtype=float)
        x2_arr = 1.0 - x1_arr
        
        gamma1_arr = np.array(solv1_gam, dtype=float)
        gamma2_arr = np.array(solv2_gam, dtype=float)
        
        ln_gam1 = np.log(gamma1_arr)
        ln_gam2 = np.log(gamma2_arr)
        
        g_excess = (x1_arr * ln_gam1) + (x2_arr * ln_gam2)
        
        term1 = np.zeros_like(x1_arr)
        mask1 = x1_arr > 1e-9
        term1[mask1] = x1_arr[mask1] * np.log(x1_arr[mask1])
        
        term2 = np.zeros_like(x2_arr)
        mask2 = x2_arr > 1e-9
        term2[mask2] = x2_arr[mask2] * np.log(x2_arr[mask2])
        
        g_mix = g_excess + term1 + term2
        
        return pd.DataFrame({
            "model": clean_name,
            "P": np.array(equi_p),
            "x1": x1_arr,
            "x2": x2_arr,
            "y1": np.array(y1),
            "y2": 1.0 - np.array(y1),
            "gamma1": gamma1_arr,
            "gamma2": gamma2_arr,
            "ln_gamma1": ln_gam1,
            "ln_gamma2": ln_gam2,
            "g_excess_reduced": g_excess,
            "g_mix_reduced": g_mix
        })

if __name__ == "__main__":
    start_wall = time.time()
    start_perf = time.perf_counter()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_wall))}")

    try:
        kill_all_aspen_processes()
        vle = AspenVLE()

        temperatures = [298.15, 323.15, 348.15, 373.15, 398.15]  # K

        systems = [
            # Series 1 (alcohols / water)
            ('METHANOL'     , 'WATER'),
            ('ETHANOL'      , 'WATER'),
            ('1-PROPANOL'   , 'WATER'),
            ('N-BUTANOL'    , 'WATER'),
            ('1-PENTANOL'   , 'WATER'),
            
            # Series 2 (aromatics / water)
            ('TOLUENE'      , 'WATER'),
            ('BENZENE'      , 'WATER'),
            ('CYCLOHEXANE'  , 'WATER'),

            # Common solvent / water pairs
            ('ACETONE'               , 'WATER'),
            ('ACETONITRILE'          , 'WATER'),
            ('DIMETHYL-SULFOXIDE'    , 'WATER'),
            ('N-METHYL-2-PYRROLIDONE', 'WATER'),

            # Acids with water
            ('ACETIC-ACID'  , 'WATER'),
            ('PROPIONIC-ACID', 'WATER'),
            ('FORMIC-ACID'  , 'WATER'),

            # Alcohol + solvent (organic-organic)
            ('METHANOL'     , 'ACETONE'),
            ('ETHANOL'      , 'ACETONE'),
            ('1-PROPANOL'   , 'ACETONE'),
            ('ETHANOL'      , 'BENZENE'),
            ('ETHANOL'      , 'TOLUENE'),

            # Nonpolar organics
            ('BENZENE'      , 'TOLUENE'),
            ('BENZENE'      , 'CYCLOHEXANE'),
            ('TOLUENE'      , 'CYCLOHEXANE'),

            # Polar organics / special pairs
            ('PHENOL'       , 'WATER'),
            ('GLYCEROL'     , 'WATER'),
            ('DIMETHYL-SULFOXIDE', 'METHANOL'),
            ('ACETONITRILE' , 'METHANOL'),
            ('ACETONITRILE' , 'ACETONE'),

            # Heterocycles / aromatics
            ('PYRIDINE'     , 'WATER'),
            ('ANILINE'      , 'WATER'),
            ('THIOPHENE'    , 'TOLUENE'),
            ('NITROBENZENE' , 'TOLUENE'),

            # Halogen / sulfur with aromatics
            ('CHLOROFORM'   , 'BENZENE'),
            ('CARBON-DISULFIDE', 'BENZENE'),
        ]

        models_to_run = ['NRTL',
                         'WILSON',
                         'UNIQUAC',
                         'UNIFAC',
                         'COSMOSAC'
                         ]

        print(f"Starting Batch Run for {len(systems)} systems x {len(temperatures)} temperatures...")

        for temperature in temperatures:
            print(f"\n{'='*60}")
            print(f"Temperature: {temperature} K")
            print(f"{'='*60}")

            for comp1, comp2 in systems:
                print(f"\nProcessing system: {comp1}/{comp2}...")
                for model in models_to_run:
                    try:
                        target_model = model
                        print(f"\n--- Processing {target_model} ---")

                        base_dir = os.path.join(parent_dir, f'outputs/aspen/binary_results/{target_model}')
                        csv_dir = os.path.join(base_dir, 'csv')
                        fig_dir = os.path.join(base_dir, 'figures')

                        os.makedirs(csv_dir, exist_ok=True)
                        os.makedirs(fig_dir, exist_ok=True)

                        df = vle.get_VLE_from_aspen([comp1, comp2], npoint=100, temperature=temperature, model_name=target_model)

                        csv_path = os.path.join(csv_dir, f'{comp1}_{comp2}_{target_model}_{temperature}K_vle.csv')
                        df.to_csv(csv_path, index=False)
                        print(f"Data saved to: {csv_path}")

                        plt.rcParams.update({
                            "font.family": "serif",
                            "font.serif": ["Times New Roman"],
                            "mathtext.fontset": "stix"
                        })
                        fontsize = 14
                        name1 = comp1
                        name2 = comp2
                        title_prefix = f'{comp1}/{comp2} ({target_model}) at {temperature} K'

                        fig, ax = plt.subplots(1, 3, figsize=(20, 6))

                        ax[0].plot(df['x1'], df['P'], 'b', marker='o', markersize=4, label='x (liquid)', linewidth=2.5)
                        ax[0].plot(df['y1'], df['P'], 'r', marker='o', markersize=4, label='y (vapor)', linewidth=2.5)
                        ax[0].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
                        ax[0].set_ylabel('Pressure (bar)', fontsize=fontsize)
                        ax[0].set_title(f'{title_prefix}: P-x-y Diagram', fontsize=fontsize)
                        ax[0].set_xlim(0, 1)
                        ax[0].legend(fontsize=fontsize)
                        ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

                        ax[1].plot(df['x1'], df['ln_gamma1'], 'green', marker='o', markersize=4, label=f'ln $\\gamma$ {name1}', linewidth=2.5)
                        ax[1].plot(df['x1'], df['ln_gamma2'], 'orange', marker='o', markersize=4, label=f'ln $\\gamma$ {name2}', linewidth=2.5)
                        ax[1].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
                        ax[1].set_ylabel('ln $\\gamma$', fontsize=fontsize)
                        ax[1].set_title(f'{title_prefix}: Activity Coefficients', fontsize=fontsize)
                        ax[1].set_xlim(0, 1)
                        ax[1].legend(fontsize=fontsize)
                        ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
                        ax[1].axhline(0, color='red', linewidth=1.0, linestyle='--')

                        ax[2].plot(df['x1'], df['g_mix_reduced'], 'purple', label='$g_{mix} / RT$ (Total)', linewidth=2.5)
                        ax[2].plot(df['x1'], df['g_excess_reduced'], 'k--', label='$g^E / RT$ (Excess)', linewidth=2.0, alpha=0.7)
                        ax[2].set_xlabel(f'mol frac, {name1}', fontsize=fontsize)
                        ax[2].set_ylabel('Energy ($RT$)', fontsize=fontsize)
                        ax[2].set_title(f'{title_prefix}: Gibbs Energy', fontsize=fontsize)
                        ax[2].set_xlim(0, 1)
                        ax[2].legend(loc='best', fontsize=fontsize)
                        ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
                        ax[2].axhline(0, color='black', linewidth=0.5)

                        plt.tight_layout()

                        plot_path = os.path.join(fig_dir, f'{comp1}_{comp2}_{target_model}_{temperature}K_plots.png')
                        plt.savefig(plot_path, dpi=300)
                        print(f"Plot saved to: {plot_path}")

                        plt.close(fig)
                    except Exception as e:
                        print(f"Error processing {comp1}/{comp2} with model {target_model} at {temperature} K: {e}")
                        kill_all_aspen_processes()
                        continue
    finally:
        end_wall = time.time()
        end_perf = time.perf_counter()
        elapsed = end_perf - start_perf
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        print(f"Script finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_wall))}")
        print(f"Total execution time: {hours}h {minutes}m {seconds:.2f}s ({elapsed:.2f} seconds)")