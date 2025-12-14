#!/usr/bin/env python3
"""
Quick verification script to check if identified dependencies are actually needed.
This script tries to import each dependency and shows import errors if they don't exist.
"""

import sys
import importlib
import subprocess


def check_dependency(package_name):
    """Try to import a package and return status."""
    try:
        importlib.import_module(package_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {str(e)}"


def get_installed_packages():
    """Get list of installed packages using pip."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                                capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')[2:]  # Skip header
        packages = {}
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    packages[parts[0].lower()] = parts[1]
        return packages
    except Exception as e:
        print(f"Could not get pip list: {e}")
        return {}


def main():
    # List of dependencies found by analysis (you'll update this after running the analyzer)
    dependencies_to_check = [
        'numpy',
        'scipy',
        'scikit-image',
        'pyvista',
        'porespy',
        'numpy-stl',
        'pypardiso'
    ]

    print("DEPENDENCY VERIFICATION")
    print("=" * 50)

    installed_packages = get_installed_packages()

    for dep in dependencies_to_check:
        # Try different import names (package name vs import name can differ)
        import_names = [dep, dep.replace('-', '_'), dep.replace('_', '-')]

        success = False
        for import_name in import_names:
            can_import, error_msg = check_dependency(import_name)
            if can_import:
                installed_version = installed_packages.get(dep.lower(), "unknown")
                print(f"✓ {dep:<15} (import: {import_name}) - version: {installed_version}")
                success = True
                break

        if not success:
            print(f"✗ {dep:<15} - NOT AVAILABLE")
            if dep.lower() in installed_packages:
                print(f"  (installed as {dep} but can't import)")

    print(f"\nPython version: {sys.version}")

    # Check for common import name variations
    print("\nCOMMON IMPORT NAME MAPPINGS:")
    print("-" * 30)
    mappings = {
        'scikit-image': 'skimage',
        'numpy-stl': 'stl',
        'pil': 'PIL',
        'opencv-python': 'cv2',
        'beautifulsoup4': 'bs4'
    }

    for pkg, import_name in mappings.items():
        if pkg in dependencies_to_check:
            can_import, _ = check_dependency(import_name)
            print(f"{pkg:<15} -> import {import_name} {'✓' if can_import else '✗'}")


if __name__ == "__main__":
    main()