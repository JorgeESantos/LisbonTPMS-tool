import sys, subprocess
from pathlib import Path

pyver = f"py{sys.version_info.major}{sys.version_info.minor}"
fname = Path(f"requirements-{pyver}.txt")

if fname.exists():
    print(f"Installing dependencies for {pyver}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(fname)], check=True)
else:
    print(f"No {fname} found. Falling back to pyproject.toml dependencies.")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], check=True)