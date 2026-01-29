import subprocess
import sys
import os

def setup_environment():
    """Setup virtual environment and install packages"""
    
    print("Creating Python virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'nlp_env'], check=True)
        print("[SUCCESS] Virtual environment 'nlp_env' created successfully!")
    except subprocess.CalledProcessError:
        print("Virtual environment already exists or failed to create")
    
    # Determine paths based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join('nlp_env', 'Scripts', 'pip.exe')
        python_path = os.path.join('nlp_env', 'Scripts', 'python.exe')
        activate_cmd = os.path.join('nlp_env', 'Scripts', 'activate.bat')
    else:  # Unix/Linux/Mac
        pip_path = os.path.join('nlp_env', 'bin', 'pip')
        python_path = os.path.join('nlp_env', 'bin', 'python')
        activate_cmd = 'source nlp_env/bin/activate'
    
    print("\nUpgrading pip...")
    try:
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        print("[SUCCESS] Pip upgraded")
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to upgrade pip")
    
    print("\nInstalling packages from requirements.txt...")
    try:
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        print("[SUCCESS] All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install packages: {e}")
    
    print("\nRegistering Jupyter kernel...")
    try:
        subprocess.run([python_path, '-m', 'ipykernel', 'install', '--user', 
                       '--name=nlp_env', '--display-name=NLP Environment'], check=True)
        print("[SUCCESS] Kernel registered successfully!")
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to register kernel")
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("To use the environment:")
    if os.name == 'nt':
        print("1. Run: nlp_env\\Scripts\\activate.bat")
    else:
        print("1. Run: source nlp_env/bin/activate")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Select 'NLP Environment' kernel in the notebook")
    print("4. Open: sentiment_analysis_attention.ipynb")

if __name__ == "__main__":
    setup_environment()