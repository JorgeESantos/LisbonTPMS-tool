#!/usr/bin/env python3
"""
Generate requirements.txt for a given Python version, ensuring the
best compatible package versions are selected, based on PyPI metadata.
Supports Windows, Linux, macOS. Warns if the current Python version
may be too new or incompatible with any package.
"""

import sys
import os
import json
import argparse
import requests
import platform
import re

# ------------------------------
# Packages to check
# ------------------------------
pkgs = {
    "numpy": None,
    "numpy-stl": None,
    "scipy": None,
    "porespy": None,
    "scikit-image": None,
    "pyvista": "0.40.0",  # force specific version if needed
    "pypardiso": None,
    # "trimesh": None  # optional
}

# ------------------------------
# Platform detection
# ------------------------------
def get_platform_tag():
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    py_tag = f"cp{py_major}{py_minor}"  # e.g., cp312
    abi_tag = f"cp{py_major}{py_minor}"  # simplified assumption

    if sys.platform.startswith("win"):
        arch = platform.architecture()[0]
        plat_tag = "win_amd64" if arch == "64bit" else "win32"
    elif sys.platform.startswith("linux"):
        plat_tag = "manylinux2014_x86_64"
    elif sys.platform.startswith("darwin"):
        plat_tag = "macosx_12_0_x86_64"
    else:
        plat_tag = "any"

    return py_tag, abi_tag, plat_tag

PY_TAG, ABI_TAG, PLATFORM_TAG = get_platform_tag()
CURRENT_PY = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# ------------------------------
# PyPI metadata query
# ------------------------------
def get_requires_python(pkg, version=None):
    """Return the requires_python string for a package/version from PyPI."""
    url = f"https://pypi.org/pypi/{pkg}/json" if version is None else f"https://pypi.org/pypi/{pkg}/{version}/json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    info = r.json()["info"]
    return info.get("requires_python") or "(none)"

def get_best_wheel(pkg, version=None):
    """Return the latest compatible version for the current platform."""

    url = f"https://pypi.org/pypi/{pkg}/json" if version is None else f"https://pypi.org/pypi/{pkg}/{version}/json"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    releases = data["releases"]

    # Sort versions descending
    sorted_versions = sorted(releases.keys(), reverse=True)
    for ver in sorted_versions:
        files = releases[ver]
        for f in files:
            if f["filename"].endswith(".whl"):
                return ver
    return version or "(source-only)"

# ------------------------------
# Version comparison helpers
# ------------------------------
def parse_requires_python(req_str):
    """Return a tuple (min_version, max_version) if possible from requires_python string."""

    if req_str in (None, "(none)"):
        return None, None
    # Very simplified: only supports >=, <=, >, < conditions
    min_v = None
    max_v = None
    parts = re.split(r",\s*", req_str)
    for p in parts:
        m = re.match(r"(>=|>|<=|<)\s*([\d\.]+)", p)
        if m:
            op, v = m.groups()
            if op in (">=", ">"):
                if not min_v or tuple(map(int, v.split("."))) > tuple(map(int, min_v.split("."))):
                    min_v = v
            elif op in ("<=", "<"):
                if not max_v or tuple(map(int, v.split("."))) < tuple(map(int, max_v.split("."))):
                    max_v = v
    return min_v, max_v

def check_python_compatibility(pkg, requires_py): #Intakes parse_requires_python()
    """Print a warning if CURRENT_PY is outside the range specified by requires_python."""

    min_v, max_v = parse_requires_python(requires_py)
    cur = tuple(sys.version_info[:3])
    if min_v and cur < tuple(map(int, min_v.split("."))):
        print(f"⚠ WARNING: Current Python {CURRENT_PY} is lower than {pkg}'s minimum required {min_v}")
    if max_v and cur > tuple(map(int, max_v.split("."))):
        print(f"⚠ WARNING: Current Python {CURRENT_PY} is higher than {pkg}'s maximum supported {max_v}")

# ------------------------------
# Generate requirements
# ------------------------------
def generate_requirements(pkgs_dict): #Intakes check_python_compatibility(), get_best_wheel(), get_requires_python()
    """Return a list of dependencies pinned to best versions for current Python/OS."""
    requirements = []
    for pkg, ver in pkgs_dict.items():
        try:
            best_ver = get_best_wheel(pkg, ver)
            requires_py = get_requires_python(pkg, best_ver)
            check_python_compatibility(pkg, requires_py)
            if best_ver:
                requirements.append(f"{pkg}>={best_ver}")
            else:
                requirements.append(pkg)
        except Exception as e:
            print(f"Warning: could not get version info for {pkg}: {e}", file=sys.stderr)
            requirements.append(pkg)
    return requirements

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate requirements.txt for current Python version")
    parser.add_argument("--in", dest="input_file", default="requirements.in", help="Input requirements.in file")
    parser.add_argument("--out", dest="output_file", default="requirements.txt", help="Output requirements.txt file")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        print(f"Reading base packages from {args.input_file}")
        #try:
        #with open(args.input_file, encoding="utf-8") as f:
        with open(args.input_file) as f:
            for line in f:
                print(line)
                line = line.strip()
                if line and not line.startswith("#"):
                    if line not in pkgs:
                        pkgs[line] = None
        #except Exception as e:
            #print

    reqs = generate_requirements(pkgs)
    with open(args.output_file, "w") as f:
        for r in reqs:
            f.write(r + "\n")

    print(f"Generated {args.output_file} with {len(reqs)} packages.")

if __name__ == "__main__":
    main()

