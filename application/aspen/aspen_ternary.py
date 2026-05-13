import win32com.client as win32
import os
import pandas as pd
import numpy as np
import psutil
import time
import matplotlib.pyplot as plt
import mpltern
from tqdm import tqdm
import sys
import gc

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

def clean_aspen_junk(folder):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and not f.lower().endswith(".bkp"):
            try:
                os.remove(path)
            except:
                pass

class AspenTernary:
    
    def _generate_simplex_grid(self, steps=60):
        grid_points = []
        min_frac = 1e-9
        denominator = steps - 1

        for i in range(steps):
            for j in range(steps - i):
                x = i / denominator
                y = j / denominator
                z = 1.0 - x - y

                if x < min_frac: x = min_frac
                if y < min_frac: y = min_frac
                if z < min_frac: z = min_frac
                
                grid_points.append([x, y, z])
        
        arr = np.array(grid_points)
        arr_unique = np.unique(np.round(arr, decimals=10), axis=0)
        return arr_unique

    def get_ternary_data(self, 
                         chemical_list, 
                         bkp_path,
                         stream_name='FEED', 
                         steps=60, 
                         temperature=298.15,
                         pressure=1.013):
        
        aspen = win32.Dispatch('Apwn.Document')
        aspen.InitFromArchive2(bkp_path)
        aspen.Visible = 0  
        aspen.SuppressDialogs = 1 
        
        try:
            assigned_comps = []
            
            for idx, comp_name in enumerate(chemical_list, 1):
                solv_id = f'SOLV{idx}'
                node_path = f'/Data/Components/Specifications/Input/DBNAME1/{solv_id}'
                aspen.Tree.FindNode(node_path).Value = comp_name.upper()
                assigned_comps.append(comp_name.upper())
            
            aspen.Reinit()
            time.sleep(2)
            
            for idx, assigned_comp in enumerate(assigned_comps, 1):
                solv_id = f'SOLV{idx}'
                node_path = f'/Data/Components/Specifications/Input/DBNAME1/{solv_id}'
                if aspen.Tree.FindNode(node_path).Value is None:
                    aspen.Tree.FindNode(node_path).Value = assigned_comp

            stream_input = aspen.Tree.FindNode(f"/Data/Streams/{stream_name}/Input")
            stream_input.FindNode("TEMP/MIXED").Value = temperature
            stream_input.FindNode("PRES/MIXED").Value = pressure
            stream_input.FindNode("TOTFLOW/MIXED").Value = 1.0 

            mole_fracs = self._generate_simplex_grid(steps)
            results = []
            total_points = len(mole_fracs)
            
            path_flow_base = f"/Data/Streams/{stream_name}/Input/FLOW/MIXED"
            node_x1 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV1")
            node_x2 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV2")
            node_x3 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV3")

            path_gamma = f"/Data/Streams/{stream_name}/Output/GAMMA/MIXED/LIQUID"
            gamma_node1, gamma_node2, gamma_node3 = None, None, None

            for i, (x, y, z) in enumerate(tqdm(mole_fracs, desc="Processing points", unit="pt")):

                if i > 0 and i % 200 == 0:
                    aspen.Close()
                    time.sleep(1)
                    kill_all_aspen_processes()
                    time.sleep(1)
                    
                    aspen = win32.Dispatch('Apwn.Document')
                    aspen.InitFromArchive2(bkp_path)
                    aspen.Visible = 0
                    aspen.SuppressDialogs = 1
                    time.sleep(1)
                    
                    for idx, assigned_comp in enumerate(assigned_comps, 1):
                        solv_id = f'SOLV{idx}'
                        aspen.Tree.FindNode(f'/Data/Components/Specifications/Input/DBNAME1/{solv_id}').Value = assigned_comp
                    
                    aspen.Reinit()
                    time.sleep(2)
                    
                    for idx, assigned_comp in enumerate(assigned_comps, 1):
                        solv_id = f'SOLV{idx}'
                        node_path = f'/Data/Components/Specifications/Input/DBNAME1/{solv_id}'
                        if aspen.Tree.FindNode(node_path).Value is None:
                            aspen.Tree.FindNode(node_path).Value = assigned_comp

                    node_x1 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV1")
                    node_x2 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV2")
                    node_x3 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV3")
                    gamma_node1 = aspen.Tree.FindNode(f"{path_gamma}/SOLV1")
                    gamma_node2 = aspen.Tree.FindNode(f"{path_gamma}/SOLV2")
                    gamma_node3 = aspen.Tree.FindNode(f"{path_gamma}/SOLV3")

                try:
                    node_x1.Value = x
                    node_x2.Value = y
                    node_x3.Value = z
                    
                    aspen.Engine.Run2()

                    if gamma_node1 is None:
                        gamma_node1 = aspen.Tree.FindNode(f"{path_gamma}/SOLV1")
                        gamma_node2 = aspen.Tree.FindNode(f"{path_gamma}/SOLV2")
                        gamma_node3 = aspen.Tree.FindNode(f"{path_gamma}/SOLV3")

                    g1 = gamma_node1.Value
                    g2 = gamma_node2.Value
                    g3 = gamma_node3.Value

                    if g1 is None or g2 is None or g3 is None:
                        raise ValueError(f"Gamma calculation failed")
                    
                    results.append([g1, g2, g3])

                    if i % 50 == 0:
                        gc.collect()

                except Exception as inner_e:
                    try:
                        aspen.Close()
                    except:
                        pass
                    kill_all_aspen_processes()
                    time.sleep(1)
                    
                    aspen = win32.Dispatch('Apwn.Document')
                    aspen.InitFromArchive2(bkp_path)
                    aspen.Visible = 0
                    aspen.SuppressDialogs = 1
                    
                    for idx, assigned_comp in enumerate(assigned_comps, 1):
                        aspen.Tree.FindNode(f'/Data/Components/Specifications/Input/DBNAME1/SOLV{idx}').Value = assigned_comp
                    
                    aspen.Reinit()
                    time.sleep(1)
                    
                    node_x1 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV1")
                    node_x2 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV2")
                    node_x3 = aspen.Tree.FindNode(f"{path_flow_base}/SOLV3")
                    gamma_node1, gamma_node2, gamma_node3 = None, None, None
                    
                    results.append([np.nan, np.nan, np.nan])

        except Exception as e:
            aspen.Close()
            raise e
            
        aspen.Close()
        
        gammas = np.array(results)
        
        valid_mask = ~np.isnan(gammas).any(axis=1)
        gammas = gammas[valid_mask]
        mole_fracs = mole_fracs[valid_mask]

        ln_gamma1 = np.log(gammas[:, 0] + 1e-9)
        ln_gamma2 = np.log(gammas[:, 1] + 1e-9)
        ln_gamma3 = np.log(gammas[:, 2] + 1e-9)

        x1, x2, x3 = mole_fracs[:, 0], mole_fracs[:, 1], mole_fracs[:, 2]

        g_excess = (x1 * ln_gamma1) + (x2 * ln_gamma2) + (x3 * ln_gamma3)
        
        x_safe = mole_fracs + 1e-9
        g_ideal = (mole_fracs * np.log(x_safe)).sum(axis=1)
        g_mix = g_excess + g_ideal
        
        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3,
            'ln_gamma1': ln_gamma1, 'ln_gamma2': ln_gamma2, 'ln_gamma3': ln_gamma3,
            'g_mix_reduced': g_mix, 'g_excess_reduced': g_excess
        })

if __name__ == "__main__":
    try:
        overall_start = time.time()
        kill_all_aspen_processes()
        ternary = AspenTernary()

        systems = [
            ("CHLOROFORM", "ACETONE", "METHANOL")
        ]

        method = "cosmosac"
        method_map = {
            "cosmosac"  : "COSMO-SAC",
            "nrtl"      : "NRTL",
            "unifac"    : "UNIFAC",
            "uniquac"   : "UNIQUAC",
            "wilson"    : "Wilson",
        }

        method_label = method_map.get(method, method.upper())

        stream_name = "FEED"
        temperatures = [398.15]
        pressure = 1.013
        steps = 61

        ternary_dir = "application/aspen/aspen_bkp_files/ternary_analysis/"
        base_bkp = os.path.abspath(os.path.join(parent_dir, f"application/aspen/aspen_bkp_files/ternary_analysis/tern_{method}.bkp"))
        base_dir = os.path.join(parent_dir, "outputs/aspen/ternary_results")
        method_dir = os.path.join(base_dir, method_label)
        os.makedirs(method_dir, exist_ok=True)
        csv_dir = os.path.join(method_dir, "csv")
        fig_dir = os.path.join(method_dir, "figures")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        for comp1, comp2, comp3 in systems:
            for temperature in temperatures:
                try:
                    comps = [comp1, comp2, comp3]
                    print(f"Running {comp1}/{comp2}/{comp3} at {temperature} K...")

                    df = ternary.get_ternary_data(
                        comps, base_bkp, stream_name, steps, temperature, pressure
                    )

                    csv_path = os.path.join(csv_dir, f"{comp1}_{comp2}_{comp3}_{temperature}K_ternary.csv")
                    df.to_csv(csv_path, index=False)

                    if mpltern:
                        plt.rcParams.update({
                            "font.family": "serif",
                            "font.serif": ["Times New Roman"],
                            "mathtext.fontset": "stix",
                            "font.size": 14
                        })
                        t, l, r = df["x1"], df["x2"], df["x3"]
                        g_excess = df["g_excess_reduced"]
                        g_mix    = df["g_mix_reduced"]

                        fig = plt.figure(figsize=(18, 8))

                        def setup_ternary(ax, t, l, r, data, names, cbar_label):
                            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
                            cntr = ax.tricontourf(t, l, r, data, levels=30, cmap='RdBu_r')
                            ax.tricontour(t, l, r, data, levels=30, colors='k', linewidths=0.3, alpha=0.5)
                            off = 0.10
                            ax.text(-off, 0.5 + off/2, 0.5 + off/2, names[0],
                                    fontsize=12, ha='center', va='top', rotation=0)
                            ax.text(0.5 + off/2, 0.5 + off/2, -off, names[1],
                                    fontsize=12, ha='center', va='bottom', rotation=60)
                            ax.text(0.5 + off/2, -off, 0.5 + off/2, names[2],
                                    fontsize=12, ha='center', va='bottom', rotation=-60)
                            cbar = plt.colorbar(cntr, ax=ax, shrink=0.7, pad=0.1)
                            cbar.set_label(cbar_label, rotation=270, labelpad=20)

                        ax1 = fig.add_subplot(121, projection="ternary")
                        setup_ternary(ax1, t, l, r, g_excess, comps, 'Excess Gibbs Energy ( $g^E / RT$ )')

                        ax2 = fig.add_subplot(122, projection="ternary")
                        setup_ternary(ax2, t, l, r, g_mix, comps, 'Gibbs Energy of Mixing ( $\\Delta_{mix}g / RT$ )')

                        plt.suptitle(f"{comp1} / {comp2} / {comp3} ({method_label}) at {temperature} K", y=0.95)

                        fig_path = os.path.join(fig_dir, f"{comp1}_{comp2}_{comp3}_{temperature}K_plots.png")
                        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                        plt.close(fig)

                except Exception as e:
                    print(f"Error in {comp1}/{comp2}/{comp3} at {temperature} K: {e}")
                    import traceback
                    traceback.print_exc()

        total = time.time() - overall_start
        print(f"Total execution time: {total:.2f} s")
    finally:
        clean_aspen_junk(ternary_dir)
        print("Cleaned Aspen junk.")