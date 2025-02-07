from .mesh_functions import mesh_from_array, Mesh_Volume
from .im_seg_functions import norm_array, trim_floating_artifacts
import numpy as np
import scipy.optimize as spot

def gradient(domain, initial_value, final_value, f=lambda x, y, z: x):
    """This function intakes a domain from the TPMS object like tpms.domain, upon which
    the gradient is supposed to by applied, calculates the
    gradient between provided values, according to the f tendency.

    This works for both porosity and/or cell size gradients."""

    f = f(domain[0], domain[1], domain[2])
    return (final_value - initial_value) * norm_array(f) + initial_value

def STL_Poro_finder(f, target_porosity, dimensions=(1.0, 1.0, 1.0), level=None, step_size=1.0, mask=None, im_seed=lambda f, c: np.where((f >= -c) & (f <= c), True, False),
                    trim_artifacts=True, x0=None, bracket=None):
    """This new function takes a mutable/or not grid (f) as input and computes
    the level-set variable c variable that would allow for the mesh porosity control."""

    print('Searching for level set value (c).\n')
    def level_set(c):
        im = im_seed(f, c).astype(bool)
        """if trim_artifacts:
            im = trim_floating_artifacts(im)"""
        #region Volume processing
        if np.amax(im) > 0:
            try:
                if mask is not None:
                    #Processing mask total volume
                    verts0, faces0, _ = mesh_from_array(im=mask(np.ones_like(im).astype(bool)), dimensions=dimensions, level=level, step_size=step_size)
                    maskV = Mesh_Volume(vertices=verts0, faces=faces0)
                    del verts0
                    del faces0
                    #Processing im volume
                    im = mask(im)
                    if trim_artifacts:
                        im = trim_floating_artifacts(im)
                    vertices, faces, normals = mesh_from_array(im=im, dimensions=dimensions, level=level, step_size=step_size)
                    del im
                    STL_P = (maskV - Mesh_Volume(vertices, faces)) / maskV
                    if STL_P < 0.0:
                        STL_P = 0.0
                    return STL_P
                else:
                    if trim_artifacts:
                        im = trim_floating_artifacts(im)
                    #Vt = np.prod(dimensions)
                    vertices, faces, normals = mesh_from_array(im=im, dimensions=dimensions, level=level, step_size=step_size)
                    #Vt = max(vertices[:, 0]) * max(vertices[:, 1]) * max(vertices[:, 2])
                    #STL_P = (Vt - Mesh_Volume(vertices, faces)) / Vt
                    del im
                    STL_P = (np.prod(dimensions) - Mesh_Volume(vertices, faces)) / np.prod(dimensions)
                    if STL_P < 0.0:
                        STL_P = 0.0
                    return STL_P
            except Exception as e:
                print(f"An error occurred: {e}\n")
                return 1.0
        else:
            return 1.0
        #endregion

    #region Optimization function
    if bracket is None:
        bracket = [np.amin(f), np.amax(f)]
    if x0 is None:
        x0 = np.mean(bracket)
    try:
        result = spot.root_scalar(lambda c: level_set(c) - target_porosity, x0=x0,
                                  method='brentq',
                                  xtol=1e-5,
                                  bracket=bracket)
        print(f'Level set value found: c={round(result.root, 6)}\n')
        return result.root
    except TypeError:
        return 0.0
    #endregion

def im_Poro_finder(f, target_porosity, mask=None, im_seed=lambda f, c: np.where((f >= -c) & (f <= c), True, False),
                   trim_artifacts=True, bracket=None, x0=None):
    """This new function takes a mutable/or not grid (f) as input and computes
    the level-set variable c variable that would allow for the image porosity control."""

    print('Searching for level set value (c).\n')
    def objective(c):
        im = im_seed(f, c)
        if np.amax(im) > 0:  # if image is not empty
            if mask is not None:
                Vt = np.count_nonzero(mask(np.ones_like(im).astype(bool)))
                im = mask(im)
                if trim_artifacts:
                    im = trim_floating_artifacts(im)
                Vf = (Vt - np.count_nonzero(im)) / Vt
                if Vf < 0.0:
                    Vf = 0.0
            else:
                if trim_artifacts:
                    im = trim_floating_artifacts(im)
                Vf = (np.prod(im.shape) - np.count_nonzero(im)) / np.prod(im.shape)
                if Vf < 0.0:
                    Vf = 0.0
        else:  # if image is empty
            Vf = 1.0
        return Vf

    #region Otimization function
    if bracket is None:
        bracket = [np.amin(f), np.amax(f)]
    if x0 is None:
        x0 = np.mean(bracket)
    try:
        result = spot.root_scalar(lambda c: objective(c) - target_porosity, x0=x0,
                                  method='brentq',
                                  xtol=1e-6,
                                  bracket=bracket)
        print(f'Level set value found: c={round(result.root, 6)}\n')
        return result.root
    except TypeError:
        return 0.0
    #endregion

def TPMS_Hybridize0(tpms1, tpms2, p=0.5, sigma=lambda x, y, z, k: 1.0 / (1.0 + np.exp(k * x)), k=2.0):
    """This function intakes two instances from TPMS,
    a sigmoid function according to which the hybridization is performed and
    a fraction (p) value that determines the axial point (as a fraction of the axis) where the transition is happening."""

    norm_domain = lambda x, p=0.5: (x + abs(np.amin(x))) / 2 - p * np.pi

    H_grid = sigma(norm_domain(tpms1.domain[0], p), norm_domain(tpms1.domain[1], p), norm_domain(tpms1.domain[2], p),
                   k) * tpms1.grid + (1 - sigma(norm_domain(tpms2.domain[0], p), norm_domain(tpms2.domain[1], p),
                               norm_domain(tpms2.domain[2], p), k)) * tpms2.grid

    """#todo se o hybridize conseguisse fazer o hybrid dos level sets?

    H_level_set = lambda f, c: (
            sigma(norm_domain(tpms1.domain[0], p), norm_domain(tpms1.domain[1], p), norm_domain(tpms1.domain[2], p), k)
            * tpms1.im_seed(f, c)
            + (
                    1 - sigma(
                norm_domain(tpms2.domain[0], p),
                norm_domain(tpms2.domain[1], p),
                norm_domain(tpms2.domain[2], p),
                k)) * tpms2.im_seed(f, c))"""

    return H_grid

def TPMS_Hybridize(tpms1, tpms2, p=0.5, sigma=lambda x, y, z, k: 1.0 / (1.0 + np.exp(k * x)), k=2.0):
    """This function intakes two instances from TPMS,
    a sigmoid function according to which the hybridization is performed and
    a fraction (p) value that determines the axial point (as a fraction of the axis) where the transition is happening."""

    norm_domain = lambda x, p=0.5: (x + abs(np.amin(x))) / 2 - p * np.amax(x)

    #Hybridization function
    Hybrid_trans = lambda x1, x2: sigma(norm_domain(tpms1.domain[0], p), norm_domain(tpms1.domain[1], p), norm_domain(tpms1.domain[2], p),
                   k) * x1 + (1.0 - sigma(norm_domain(tpms2.domain[0], p), norm_domain(tpms2.domain[1], p),
                                                norm_domain(tpms2.domain[2], p), k)) * x2

    """The user can use the Hybrid_trans function to perform the transition between TPMS.grids, offset c values
     and the correponding level-set conditions with special attention to this one regarding possible overlapping of the
     provided level-set conditions."""

    return Hybrid_trans, sigma