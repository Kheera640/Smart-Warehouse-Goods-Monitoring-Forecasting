import subprocess
import sys

packages = [
    "pandas",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "flask",
    "python-dotenv",
    "joblib"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
