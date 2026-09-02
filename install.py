import subprocess
import sys
import shutil

PYTHON_VER = "3.11"
BASE_CONDA_PKGS = ["pandas", "matplotlib", "mpltern",
                   "seaborn", "scikit-learn",
                   "networkx", "rdkit", "jupyter", "ipykernel"]
BASE_PIP_PKGS   = ["optuna", "optuna-dashboard"]
TORCH_PKGS      = ["torch==2.8.0",
                   "torchvision==0.23.0",
                   "torchaudio==2.8.0"]
PYG_PKGS        = ["torch_geometric", "pyg_lib", "torch_scatter", 
                   "torch_sparse", "torch_cluster", "torch_spline_conv"]

def run_command(command, desc):
    print(f"\n--- {desc} ---")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        print(f"Error during: {desc}")
        sys.exit(1)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available using nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        subprocess.check_call([nvidia_smi], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("starting installation...")
    
    # ASK USER FOR VERSION
    while True:
        choice = input("\nselect installation type:\n  [1] GPU (CUDA)\n  [2] CPU\nenter choice (1 or 2): ").strip()
        if choice == "1":
            use_gpu = True
            break
        elif choice == "2":
            use_gpu = False
            break
        else:
            print("invalid choice. please enter 1 or 2.")

    if use_gpu:
        # Check for NVIDIA GPU
        if not check_nvidia_gpu():
            raise RuntimeError(
                "GPU installation selected, but no NVIDIA GPU was detected.\n"
                "possible reasons:\n"
                "  1. no NVIDIA GPU is installed on this system.\n"
                "  2. NVIDIA drivers are not installed or not properly configured.\n"
                "  3. 'nvidia-smi' is not in your system PATH.\n"
                "\npossible solutions:\n"
                "  - install NVIDIA drivers from: https://www.nvidia.com/drivers\n"
                "  - make sure CUDA Toolkit and cuDNN are installed.\n"
                "  - or select CPU installation (option 2) instead."
            )
        
        # GPU
        env_name = "nonisothermal-deepthermomix-gpu"
        conda_pkgs = ["nomkl"] + BASE_CONDA_PKGS # add nomkl
        torch_flags = ["--index-url", "https://download.pytorch.org/whl/cu129"]
        pyg_flags = ["-f", "https://data.pyg.org/whl/torch-2.8.0+cu129.html"]
        
    else:
        # CPU
        env_name = "nonisothermal-deepthermomix-cpu"
        conda_pkgs = ["nomkl"] + BASE_CONDA_PKGS # add nomkl
        torch_flags = [] # Default PyPI for CPU
        pyg_flags = ["-f", "https://data.pyg.org/whl/torch-2.8.0+cpu.html"]

    cmd_create = ["conda", "create", "-n", env_name, f"python={PYTHON_VER}", "pip"] + conda_pkgs + ["-c", "conda-forge", "-y"]
    run_command(cmd_create, f"Creating Env: {env_name}")
    pip_cmd = ["conda", "run", "-n", env_name, "python", "-m", "pip", "install"]
    run_command(pip_cmd + TORCH_PKGS + torch_flags, "installing torch stack...")
    run_command(pip_cmd + PYG_PKGS + pyg_flags, "installing graph neural network stack...")
    run_command(pip_cmd + BASE_PIP_PKGS, "installing Optuna...")
    run_command(pip_cmd + ["-e", "."], "installing 'nonisothermal-deepthermomix' package...")

    print(f"\nINSTALLATION COMPLETE! Environment: {env_name}")
    print(f"to start:  conda activate {env_name}")

if __name__ == "__main__":
    main()