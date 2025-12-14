"""The objective of this script is to determine the minimum Python verison needed to run the code"""

import sys, requests

pkgs = {
    "numpy": None,
    "numpy-stl": None,
    "scipy": None,
    "porespy": None,
    "scikit-image": None,
    "pyvista": "0.40.0",   # specify version if you want to check a particular release
    "pypardiso": None,
    # "trimesh": None   # optional
}

install_requires=[
    "numpy",
    "numpy-stl",
    "scipy",
    "porespy",
    "scikit-image",
    "pyvista>=0.40,<0.50",
    "pypardiso",
]

def get_requires_python(pkg, version=None):
    url = f"https://pypi.org/pypi/{pkg}/json" if version is None else f"https://pypi.org/pypi/{pkg}/{version}/json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    info = r.json()["info"]
    return info.get("requires_python") or "(none)"

for pkg, ver in pkgs.items():
    try:
        rp = get_requires_python(pkg, ver)
        print(f"{pkg}{'=='+ver if ver else ''}: requires_python = {rp}")
    except Exception as e:
        print(f"{pkg}: ERROR: {e}", file=sys.stderr)