# SETUP

In PowerShell run the following command:

.\setup_env.ps1

Or do the following to setup manually

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# RUN

python run.py