# 1. Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# 2. Activate environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# 4. Install dependencies
Write-Host "Installing project dependencies..."
pip install opencv-python mediapipe numpy

Write-Host "`n=== Environment setup complete! ==="
Write-Host "Activate it anytime by running: .\venv\Scripts\Activate.ps1"