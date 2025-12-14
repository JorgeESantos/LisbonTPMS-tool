#!/usr/bin/env python3
"""
gen_matrix_top_pkgs.py

Focuses on a fixed top-level pkgs list (possibly with constraints) and generates
requirements-py<maj><min>.txt files for Python minors that are fully supported.

Usage:
  python gen_matrix_top_pkgs.py \
    --requirements requirements.txt \
    --py "3.10,3.11,3.12"

If a top-level package is not pinned in requirements.txt, the script will query
PyPI and choose the newest release matching the constraint (if any).
"""

from pathlib import Path
import argparse
import sys
import re
import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion
import platform

# -------------------------
# CONFIGURE your top-level pkgs here
# -------------------------
pkgs = [
    "numpy",
    "numpy-stl",
    "scipy",
    "porespy",
    "scikit-image",
    "pyvista>=0.40,<0.50",
    "pypardiso",
]

# -------------------------
# Platform detection
# -------------------------
def detect_platform_tag():
    plat = sys.platform
    arch = platform.machine().lower()
    if plat.startswith("win"):
        return "win_amd64" if "64" in arch or arch in ("amd64", "x86_64") else "win32"
    if plat.startswith("linux"):
        return "manylinux"
    if plat.startswith("darwin"):
        return "macosx"
    return "any"

PLATFORM = detect_platform_tag()

# -------------------------
# Helpers
# -------------------------
def split_name_and_constraint(spec: str):
    """
    Return (name, specifier_str) from strings like 'pyvista>=0.40,<0.50' or 'numpy'
    """
    m = re.match(r"^\s*([A-Za-z0-9_\-\.]+)\s*(.*)$", spec)
    if not m:
        return spec.strip(), ""
    name = m.group(1)
    rest = m.group(2).strip()
    return name, rest

def read_pinned_requirements(requirements_path: Path):
    """Return dict name->version for pinned lines in requirements.txt"""
    pins = {}
    if not requirements_path.exists():
        return pins
    text = requirements_path.read_text(encoding="utf-8")
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "==" in ln:
            name, ver = ln.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins

def releases_for_package(pkg: str):
    """Return release dict from PyPI (name case-sensitive issues: use pypi endpoint)"""
    url = f"https://pypi.org/pypi/{pkg}/json"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"PyPI query failed for {pkg}: {r.status_code}")
    return r.json().get("releases", {})

def choose_best_version_from_pypi(pkg: str, constraint: str = ""):
    """
    Return the newest Version string from PyPI releases that satisfies SpecifierSet(constraint).
    If constraint is empty, returns the latest final release.
    """
    try:
        releases = releases_for_package(pkg)
    except Exception as e:
        raise RuntimeError(f"Could not fetch releases for {pkg}: {e}")

    spec = SpecifierSet(constraint) if constraint else None
    # collect valid versions
    versions = []
    for v in releases.keys():
        try:
            vv = Version(v)
        except InvalidVersion:
            continue
        # skip pre-releases unless explicitly allowed by specifier
        if not spec:
            # include pre-releases? prefer final releases, but keep them if no better option
            versions.append(vv)
        else:
            if vv in spec:
                versions.append(vv)
    if not versions:
        return None
    versions.sort(reverse=True)
    # prefer final releases first. Return highest final release if present, else highest pre-release.
    finals = [v for v in versions if not v.is_prerelease]
    return str(finals[0] if finals else versions[0])

def release_files_for(pkg: str, version: str):
    """Return list of file dicts for a specific release version from PyPI"""
    url = f"https://pypi.org/pypi/{pkg}/{version}/json"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return []
    data = r.json()
    # prefer version-specific releases
    files = data.get("releases", {}).get(version, [])
    if not files:
        # fallback to urls (these usually contain files for the exact version)
        files = data.get("urls", [])
    return files

def wheel_is_compatible(filename: str, py_major: int, py_minor: int):
    """
    Permissive check for filename compatibility:
    Accept py3-none-any, abi3, none-any, py3 wheels, cp{maj}{min}, cp{maj}
    """
    f = filename.lower()
    if not f.endswith(".whl"):
        return False
    if "none-any.whl" in f or "py3-none-any.whl" in f:
        return True
    if "abi3" in f:
        return True
    cp_tag = f"cp{py_major}{py_minor}"
    if cp_tag in f:
        return True
    if f"cp{py_major}" in f:
        return True
    if f"py{py_major}{py_minor}" in f:
        return True
    return False

# -------------------------
# Main check routine
# -------------------------
def check_top_pkgs_and_emit(pinned_req_path: Path, top_pkgs, py_versions):
    pinned = read_pinned_requirements(pinned_req_path)
    # map top-level name -> (constraint_str)
    parsed = []
    for s in top_pkgs:
        name, constraint = split_name_and_constraint(s)
        parsed.append((name, constraint))

    # Determine chosen version for each top-level package (prefer pinned)
    chosen = {}  # name -> version_str
    diagnostics = []
    for name, constraint in parsed:
        if name in pinned:
            chosen_ver = pinned[name]
            diagnostics.append(f"Using pinned {name}=={chosen_ver} from {pinned_req_path.name}")
        else:
            # query PyPI for best version that satisfies constraint
            chosen_ver = choose_best_version_from_pypi(name, constraint)
            if chosen_ver:
                diagnostics.append(f"Selected {name}=={chosen_ver} from PyPI to satisfy '{constraint or 'any'}'")
            else:
                diagnostics.append(f"ERROR: Could not find any release on PyPI for {name} matching '{constraint or 'any'}'")
        chosen[name] = chosen_ver

    # now check wheel availability per python version
    results = {}  # pystr -> (supported_bool, missing_list)
    for maj, mino in py_versions:
        pystr = f"{maj}.{mino}"
        missing = []
        for name, version in chosen.items():
            if not version:
                missing.append((name, None, "no version chosen"))
                continue
            files = release_files_for(name, version)
            ok = False
            seen_fnames = []
            for f in files:
                fname = f.get("filename", "")
                seen_fnames.append(fname)
                if wheel_is_compatible(fname, maj, mino):
                    ok = True
                    break
            if not ok:
                missing.append((name, version, seen_fnames))
        results[pystr] = (len(missing) == 0, missing)

    # Emit results and write files where supported
    for pystr, (supported, missing) in results.items():
        outname = f"requirements-py{pystr.replace('.','')}.txt"
        if supported:
            # write a copy of pinned requirements.txt (if exists) or construct minimal pinned file using chosen mapping
            if pinned_req_path.exists():
                content = pinned_req_path.read_text(encoding="utf-8")
            else:
                # generate minimal pinned content from chosen
                lines = [f"{n}=={v}" for n, v in chosen.items() if v]
                content = "\n".join(lines) + "\n"
            Path(outname).write_text(content, encoding="utf-8")
            print(f"✅ Wrote {outname} (all top-level packages have wheels for Python {pystr})")
        else:
            print(f"❌ Skipping {outname}: missing wheels for Python {pystr}")
            for (name, ver, seen) in missing:
                if ver is None:
                    print(f"   - {name}: no version available/selected")
                else:
                    # give concise hint
                    print(f"   - {name}=={ver}: no compatible wheel found. Sample files: {seen[:6]}{'...' if len(seen)>6 else ''}")

    # print diagnostics
    print("\nDiagnostics:")
    for d in diagnostics:
        print(" -", d)

    return results

# -------------------------
# CLI
# -------------------------
def parse_py_list(s: str):
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "." in token:
            a, b = token.split(".", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(token), 0))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requirements", "-r", default="requirements.txt", help="Pinned requirements.txt (optional)")
    p.add_argument("--py", default="3.10,3.11,3.12", help="Comma-separated python minor versions to test (e.g. 3.10,3.11)")
    args = p.parse_args()

    # --- Locate requirements.txt automatically if not absolute ---
    pinned = Path(args.requirements)
    if not pinned.is_absolute():
        # Search one level above the script’s folder
        script_dir = Path(__file__).resolve().parent
        candidate = (script_dir.parent / pinned).resolve()
        if candidate.exists():
            pinned = candidate
        else:
            # Fall back to local path if nothing found
            pinned = Path(args.requirements).resolve()

    py_versions = parse_py_list(args.py)

    print("Platform tag (detected):", PLATFORM)
    print("Top-level pkgs to check:", pkgs)
    try:
        check_top_pkgs_and_emit(pinned, pkgs, py_versions)
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()