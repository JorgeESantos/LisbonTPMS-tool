import os, shutil, pathlib
from setuptools import find_packages, setup
from subprocess import run, CalledProcessError
from setuptools.command.install import install

desktop = pathlib.Path.home() / 'Desktop'

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

#region Prepare requirements.txt
# Function to compile requirements.in into requirements.txt using pip-tools
def compile_requirements(input_file="requirements.in", output_file="requirements.txt"):
    if os.path.exists(input_file):
        try:
            # Call pip-compile directly
            run(["pip-compile", input_file, "--output-file", output_file], check=True)
        except CalledProcessError as e:
            print(f"Error compiling requirements: {e}")
            raise
    else:
        print(f"Input file {input_file} not found. Skipping compilation.")

# Compile requirements if requirements.txt does not exist
if not os.path.exists("requirements.txt"):
    try:
        # Ensure pip-tools is installed in the environment
        run(["pip", "install", "pip-tools"], check=True)
        compile_requirements()
    except CalledProcessError as e:
        print(f"Error ensuring pip-tools is installed or compiling requirements: {e}")
        raise

# Read dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
#endregion

setup(
    name="LisbonTPMStool",
    version="1.0.0",
    description="A TPMS generator with segmentation and processing tools",
    package_dir={"LisbonTPMStool": "LisbonTPMStool"},
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Jorge E. Santos",
    author_email="jorge.e.santos@tecnico.ulisboa.pt",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=parse_requirements("requirements.txt"),
    setup_requires=["pip-tools"],
    python_requires=">=3.9",
)