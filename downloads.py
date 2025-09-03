import subprocess
import sys

packages = [
     "gradio", "PyPDF2",
    "pandas", "matplotlib", "scikit-learn", "nltk", "google-genai", "python-dotenv"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
