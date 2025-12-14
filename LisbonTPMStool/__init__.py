from .TPMS import TPMS_domain, TPMS
from . import FE_functions, im_seg_functions, mesh_functions, Utilities

"""# Define expected location for user-editable Surfaces.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes one level up from the package
USER_SURFACES_FILE = os.path.join(ROOT_DIR, "Surfaces.py")

# Check if the editable Surfaces.py exists outside the virtual environment
if os.path.exists(USER_SURFACES_FILE):
    #print(f"✔ Loading user-modified Surfaces.py from {USER_SURFACES_FILE}")
    sys.path.insert(0, ROOT_DIR)  # Ensure Python searches in the root directory first
    import Surfaces  # Import the user-modified version
else:
    #print("⚠ No user-modified Surfaces.py found in project root, using package version.")
    from . import Surfaces  # Fall back to the package version"""

__all__ = ['FE_functions', 'im_seg_functions', 'mesh_functions', 'Utilities', 'Surfaces']