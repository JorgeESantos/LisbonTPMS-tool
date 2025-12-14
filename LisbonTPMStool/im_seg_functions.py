import numpy as np
import pyvista as pv
import os, pathlib, numbers
import scipy.ndimage as spim
import skimage.morphology as skmorph
from itertools import product
from porespy.filters import local_thickness
from porespy.metrics import pore_size_distribution
from edt import edt3d
import pandas as pd

def norm_array(im):
    """Normalizes an array (can be a gray image) returning it with values between 0 and 1."""
    # With positive translation
    if np.amin(im) < 0.0:
        #trans_value = np.amin(im)  # Need this to revert to original intensity
        im = im + abs(np.amin(im))
    elif np.amin(im) > 0.0:
        im = im - abs(np.amin(im))
    # Actually lets make it between 0 to 1
    norm_im = im / abs(np.amax(im))
    return norm_im

#Slicing
def get_index_slice_bounds(object):
    """Compute orthogonal bounds of a given object as a boolean mask."""

    indices = np.argwhere(object)  # Find indices of nonzero elements
    bounds = [slice(np.amin(indices[:, i]), np.amax(indices[:, i]) + 1) for i in range(object.ndim)]
    return tuple(bounds)  # Return as tuple for easy use in slicing

def slice_array(array, bounds):
    return array[bounds]

def force_binary_fill_holes(im, threshold=float(1/8), kernel_size=3):
    """Intakes a boolean mask and fills all holes inside it that are smaller than threshold
    as a proportion of the image size."""

    #todo o threshold podia ser inferior ao size da im np.sum(count_nonzero)
    if np.amax(im) < 1:
        raise ValueError('Image is empty. There are no holes to fill.\n')
    im = (spim.binary_fill_holes(im) == 0.0) #Remove all holes that do not touch the image boundary.
    im = np.where(np.logical_xor(im, trim_floating_artifacts(im=im, threshold=threshold, kernel_size=kernel_size)), False, im) #Remove artifacts attached to the boundary.
    if np.amax(im) < 1:
        raise ValueError('force_binary_fill_holes: Error filling holes. Image became empty.\n')
    im = (spim.binary_fill_holes(im == 0.0)) #Once the boundary artifacts are removed, it ain't hurt to run fill holes again.
    print('Holes have been filled.\n')
    return im

def flood_fill_objects_size(im, kernel_size=3, rel=False):
    """Takes a binary image and labels each object with its respective number of pixels.
    This is the same as porespy.filters.region_size().

    This version only uses numpy."""
    if len(im.shape) == 2:
        strel = skmorph.square(kernel_size).astype(bool)
    elif len(im.shape) == 3:
        strel = skmorph.cube(kernel_size).astype(bool)
    regions, _ = spim.label(im, structure=strel)  # Labels everything that is non-zero
    for label in np.unique(regions[regions > 0]):
        regions = np.where(regions == label, np.count_nonzero(regions == label), regions)
    if rel:
        return regions / np.prod(im.shape)
    else:
        return regions

def get_largest_object(im):
    """Intakes a boolean mask and returns another containing the largest
     detected object inside the provided boolean mask."""

    size_regions = flood_fill_objects_size(im=im)
    if np.amax(size_regions) > 0:
        return size_regions == np.amax(size_regions)
    else:
        #print('No largest object detected.\n')
        return im

def trim_floating_artifacts(im, threshold=float(1/8), kernel_size=3):
    """Takes a binary image, identifies and labels the objects pixels by their size.

    Deletes objects smaller than threshold (as a proportion of image)."""

    strel = skmorph.cube(kernel_size).astype(bool)
    if len(im.shape) == 2:
        strel = skmorph.square(kernel_size).astype(bool)
    elif len(im.shape) == 3:
        strel = skmorph.cube(kernel_size).astype(bool)
    regions, N = spim.label(im, structure=strel)  # Labels everything that is non-zero
    # Check if the image is blank
    if (N == 0) and (np.amax(regions) < 1):
        #print('Image is blank. No objects to label.\n')
        return im
    # Check if there is a single object within image
    elif np.amax(regions) == 1.0 and N <= np.amax(regions):
        # print('Single region identified.\n')
        return im
    elif (np.amax(regions) > 1.0) and (N > 1): #if it is not empty nor is a single object
        regions = flood_fill_objects_size(im, kernel_size=kernel_size, rel=True)
    return np.where(regions > threshold, im, False)

#Array splitage
def get_split_factors(nblocks, ndim):
    """
    Returns a tuple of length `ndim` such that the product equals nblocks and the factors are as balanced as possible.
    """

    def rec(remaining, dims):
        # Base case: only one dimension left.
        if dims == 1:
            if remaining >= 1:
                return [(remaining,)]
            else:
                return []
        factors = []
        # Try all factors for the current dimension.
        for i in range(1, remaining + 1):
            if remaining % i == 0:
                for rest in rec(remaining // i, dims - 1):
                    factors.append((i,) + rest)
        return factors

    all_factors = rec(nblocks, ndim)
    # Ideal factor in each dimension for balanced blocks.
    ideal = nblocks ** (1 / ndim)

    # Score a tuple by its sum of squared deviations from the ideal.
    def score(factor_tuple):
        return sum((f - ideal) ** 2 for f in factor_tuple)

    best = min(all_factors, key=score)
    return best

def split_array(arr, nblocks):
    """
    Splits an n-dimensional array into nblocks pieces.

    Parameters:
        arr (np.ndarray): The array to split.
        nblocks (int): Total number of pieces desired.

    Returns:
        blocks (list of np.ndarray): List of blocks.
        slices_info (list of lists): For each axis, the list of slice objects used.
    """
    ndim = arr.ndim
    factors = get_split_factors(nblocks, ndim)

    # For each axis, compute slices
    slices_info = []
    for axis, splits in enumerate(factors):
        axis_len = arr.shape[axis]
        # Create indices that split the axis into 'splits' pieces.
        indices = np.linspace(0, axis_len, splits + 1, dtype=int)
        axis_slices = [slice(indices[i], indices[i + 1]) for i in range(splits)]
        slices_info.append(axis_slices)

    # Use the Cartesian product of slices for each axis to obtain blocks.
    blocks = []
    for slice_combo in product(*slices_info):
        blocks.append(arr[slice_combo])

    return blocks, slices_info

def reassemble_array(blocks, original_shape, slices_info):
    """
    Reassembles an array from blocks using the provided slice info.

    Parameters:
        blocks (list of np.ndarray): List of blocks (in the order produced by split_array).
        original_shape (tuple): The shape of the original array.
        slices_info (list of lists): List of slice objects for each axis.

    Returns:
        np.ndarray: The reassembled array.
    """
    reassembled = np.empty(original_shape, dtype=blocks[0].dtype)
    idx = 0
    for slice_combo in product(*slices_info):
        reassembled[slice_combo] = blocks[idx]
        idx += 1
    return reassembled

def compute_size_distribution(im, units=None, bins=20, log_scale=False, save_data=False, name='Sizes data', path=None):
    """Intakes an image / array with the sizes to compute the distribution over.

    Applies the PoreSpy pore_size_distribution() metric function and
    returns the size distribution in the form of the Probability Density Function (pdf %)
    and the Cumulative Density Function (cdf 1)."""

    sd = pore_size_distribution(im, bins=bins, log=log_scale, voxel_size=1.0)  #This a is a PoreSpy Results object
    if sd['pdf'][-1] != 0.0:
        sd['R'] = np.append(sd['R'], values=sd['R'][-1] + (sd['R'][-1] - sd['R'][-2]))
        sd['pdf'] = np.append(sd['pdf'], values=0.0)
        sd['cdf'] = np.append(sd['cdf'], values=0.0)
        sd['bin_widths'] = np.append(sd['bin_widths'], values=sd['bin_widths'][-1])
        sd['bin_centers'] = np.append(sd['bin_centers'], values=sd['bin_centers'][-1] + (
                sd['bin_centers'][-1] - sd['bin_centers'][-2]))
        sd['satn'] = np.append(sd['satn'], values=1.0)
    # region Convert the PoreSpy Results object to a regular dictionary
    sd = sd.__dict__
    sd = {key: value for key, value in sd.items() if not key.startswith('_')}
    if 'R' in sd:
        sd[f'D ({units})'] = sd.pop('R')
    if 'pdf' in sd:
        sd['pdf (%)'] = sd.pop('pdf')
    if 'cdf' in sd:
        sd['cdf (1)'] = sd.pop('cdf')
    if 'satn' in sd:
        sd['satn (1)'] = sd.pop('satn')
    # endregion
    #im = im * voxel_size
    print('Size distribution processed.\n')

    if save_data:
        #Converts it to an Excel file
        # Converts the dictionary into a pandas dataframe and saves to an Excel file
        df = pd.DataFrame.from_dict(sd)
        # To excel
        excel_writer = pd.ExcelWriter(path=os.path.join(path, f'{name}.xlsx'),
                                      engine='xlsxwriter')
        df.to_excel(excel_writer=excel_writer, index=False)
        excel_writer.close()
    return sd

def Local_Thickness(im, voxel_size, sizes=25, mode='hybrid', edt_black_border=False, divs=1):
    """Performs the Local Thickness algorithm implemented by PoreSpy but includes how the user wants to address
    the Euclidean Distance Transform (edt) of the foreground phase (im).

    Regardless how the user chooses to process the edt, needs to be aware that only the max value of the edt plays a
    key or impactful role on how the LT algorithm will flood the foreground.

    Returns the local thickness values as diameters at the voxel scale."""

    print('Executing Local Thickness.\n')

    # Set up sizes to use
    if isinstance(sizes, int):
        dt = edt3d(im, black_border=edt_black_border)
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0.0, num=sizes)
    del dt

    # Compute the Local Thickness
    lt = local_thickness(im, mode=mode, sizes=sizes, divs=divs) * 2.0 * voxel_size
    print('Local Thickness processed.\n')
    return lt

def MA_skel_edt(im, black_border=True):
    """im is a binary image with the foreground phase values set to 1
    Returns de edt diameters, in voxels, over the im skeleton / medial axis."""

    skel = np.where(skmorph.skeletonize(im) != 0.0, True, False)
    if black_border:
        edt = edt3d(im, black_border=True)
        skel = skel * edt * 2
    else:
        edt = edt3d(im, black_border=False)
        skel = skel * edt * 2
    return skel

def PyVista_Voxels(im, voxel_size, threshold=None, units=None, cmap='jet', opacity=1.0, camera_position=None, save_fig=False, name='TPMS', path=None):
    # region Voxel image
    grid = pv.ImageData(origin=(0, 0, 0), spacing=(voxel_size, voxel_size, voxel_size),
                        dimensions=(im.shape[0] + 1, im.shape[1] + 1, im.shape[2] + 1))
    grid.cell_data["values"] = im.flatten(order='F').astype(np.float32)
    grid = grid.threshold() #Attempt of removing NaNs
    # endregion

    if save_fig:
        # region Create a plotting window
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        pl.remove_bounds_axes()
        # region Bounds
        pl.show_bounds(bounds=[0.0, max(grid.points[:, 0]), 0.0, max(grid.points[:, 1]), 0.0, max(grid.points[:, 2])],
                       font_size=21,
                       font_family='times',
                       bold=False,
                       color='k',
                       xtitle=f'x ({units})',
                       ytitle=f'y ({units})',
                       ztitle=f'z ({units})',
                       grid=False,
                       # location='outer',
                       location='origin',
                       # location='all',
                       # use_2d=True,
                       ticks='inside',
                       show_xlabels=True,
                       show_ylabels=True,
                       show_zlabels=True,
                       n_xlabels=2,
                       n_ylabels=2,
                       n_zlabels=2,
                       minor_ticks=False,
                       fmt='%.1f',
                       padding=0.0,
                       use_3d_text=False)
        # endregion
        #region Scalar bar - create dictionary of parameters to control Scalar Bar
        sargs = dict(interactive=False,
                     title=f'Values ({units})',
                     color='black',
                     title_font_size=20,
                     label_font_size=14,
                     shadow=False,
                     n_labels=5,
                     italic=False,
                     fmt="%.3f",
                     font_family="times")
        #endregion
        if threshold is None:
            pl.add_mesh(grid.threshold(grid.get_data_range(), all_scalars=True), cmap=cmap, opacity=opacity,
                        clim=[np.amin(im[~np.isnan(im)]), np.amax(im[~np.isnan(im)])],
                        scalar_bar_args=sargs)
        else:
            pl.add_mesh(grid.threshold(threshold, all_scalars=True), cmap=cmap, opacity=opacity,
                        clim=[np.amin(im[~np.isnan(im)]), np.amax(im[~np.isnan(im)])],
                        scalar_bar_args=sargs)
        #endregion
        # Saving
        if path is not None:
            png = os.path.join(path, name + '.png')
        else:
            path = pathlib.Path.home() / 'Desktop'
            png = os.path.join(path, name + '.png')
        win_size = pl.window_size
        if camera_position is not None:
            pl.camera_position = camera_position
        pl.show(interactive=False, auto_close=True, screenshot=png, window_size=2 * win_size)
    else:
        # region Create a plotting window
        pl = pv.Plotter(off_screen=False)
        pl.set_background('white')
        pl.remove_bounds_axes()
        # region Bounds
        pl.show_bounds(bounds=[0.0, max(grid.points[:, 0]), 0.0, max(grid.points[:, 1]), 0.0, max(grid.points[:, 2])],
                       font_size=21,
                       font_family='times',
                       bold=False,
                       color='k',
                       xtitle=f'x ({units})',
                       ytitle=f'y ({units})',
                       ztitle=f'z ({units})',
                       grid=False,
                       # location='outer',
                       location='origin',
                       # location='all',
                       # use_2d=True,
                       ticks='inside',
                       show_xlabels=True,
                       show_ylabels=True,
                       show_zlabels=True,
                       n_xlabels=2,
                       n_ylabels=2,
                       n_zlabels=2,
                       minor_ticks=False,
                       fmt='%.1f',
                       padding=0.0,
                       use_3d_text=False)
        # endregion
        # region Scalar bar - create dictionary of parameters to control Scalar Bar
        sargs = dict(interactive=False,
                     title=f'Values ({units})',
                     color='black',
                     title_font_size=20,
                     label_font_size=14,
                     shadow=False,
                     n_labels=5,
                     italic=False,
                     fmt="%.3f",
                     font_family="times")
        # endregion
        #endregion
        if threshold is None:
            pl.add_mesh(grid.threshold(grid.get_data_range(), all_scalars=True), cmap=cmap, opacity=opacity, clim=[np.amin(im[~np.isnan(im)]), np.amax(im[~np.isnan(im)])],
                        scalar_bar_args=sargs)
        else:
            pl.add_mesh(grid.threshold(threshold, all_scalars=True), cmap=cmap, opacity=opacity, clim=[np.amin(im[~np.isnan(im)]), np.amax(im[~np.isnan(im)])],
                        scalar_bar_args=sargs)
        pl.show(interactive=True, auto_close=True)
    pl.close()
    pl.deep_clean()
    return grid

def PyVista_TriMeshes_plot(meshes, units='mm', camera_position=None, show_edges=True, show_bounds=True, save_fig=False, name='TPMS', path=None):
    """Mounts a 3D image with multiple triangular meshes provided as list of:

    [[vertices (ndarray), faces (ndarray), face_color (str), color_opacity (float)]]."""

    if not isinstance(meshes, list):
        raise ValueError('meshes must be a list in which each mesh must be of type list like [vertices, faces, face_color (str), opacity (float)].')

    #region Image
    pl = pv.Plotter(off_screen=False)
    pl.set_background('white')
    pl.remove_bounds_axes()
    if show_bounds:
        #region Bounds
        pl.show_bounds(
            font_size=21,
            font_family='times',
            bold=False,
            color='k',
            xtitle=f'x ({units})',
            ytitle=f'y ({units})',
            ztitle=f'z ({units})',
            grid=False,
            location='origin',
            #use_2d=True,
            ticks='inside',
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            n_xlabels = 2,
            n_ylabels = 2,
            n_zlabels = 2,
            minor_ticks=False,
            fmt='%.1f',
            padding=0.0,
            use_3d_text=False)
        #endregion

    #region Adding meshes
    if len(meshes) >= 1: #If there is one or more meshes
        for i, lista in enumerate(meshes):
            if not isinstance(lista[3], (numbers.Number, type)): #if opacity is not defined
                lista[3] = 1.0
            if not isinstance(lista[2], str): #if it has no color
                print('Uncolored mesh added.\n')
                pl.add_mesh(mesh=pv.PolyData.from_regular_faces(lista[0], lista[1]), opacity=lista[3], show_scalar_bar=False,
                            edge_color='black')
            else: #if it has color
                print('Colored mesh added.\n')
                pl.add_mesh(mesh=pv.PolyData.from_regular_faces(lista[0], lista[1]), color=lista[2], edge_color='black',
                            opacity=lista[3], show_edges=show_edges, show_scalar_bar=False)
    #endregion
    #endregion

    #region Saving and plotting
    if save_fig:
        if path is not None:
            png = os.path.join(path, name + '.png')
        else:
            path = pathlib.Path.home() / 'Desktop'
            png = os.path.join(path, name + '.png')
        win_size = pl.window_size
        if camera_position is not None:
            pl.camera_position = camera_position
        pl.show(interactive=False, auto_close=True, screenshot=png, window_size=2 * win_size)
        #pl.show(interactive=True, auto_close=True)
    else:
        if camera_position is not None:
            pl.camera_position = camera_position
        pl.show(interactive=True, auto_close=True)
    #endregion
    pl.close()
    pl.deep_clean()
    return

def PyVista_Binary_Voxels(im, voxel_size, units=None, color='oldlace', edge_color='darkgray', show_edges=False, opacity=1.0, camera_position=None, save_fig=False, path=None, name='TPMS'):
    #Building the voxel Image
    grid = pv.ImageData(origin=(0, 0, 0), spacing=(voxel_size, voxel_size, voxel_size), dimensions=(im.shape[0] + 1, im.shape[1] + 1, im.shape[2] + 1))
    grid.cell_data["values"] = im.flatten(order='F').astype(np.uint8)
    grid = grid.threshold()

    if save_fig:
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        pl.remove_bounds_axes()
        # region Bounds
        pl.show_bounds(bounds=[0.0, max(grid.points[:, 0]), 0.0, max(grid.points[:, 1]), 0.0, max(grid.points[:, 2])],
                       font_size=21,
                       font_family='times',
                       bold=False,
                       color='k',
                       xtitle=f'x ({units})',
                       ytitle=f'y ({units})',
                       ztitle=f'z ({units})',
                       grid=False,
                       location='origin',
                       ticks='inside',
                       show_xlabels=True,
                       show_ylabels=True,
                       show_zlabels=True,
                       n_xlabels=2,
                       n_ylabels=2,
                       n_zlabels=2,
                       minor_ticks=False,
                       fmt='%.1f',
                       padding=0.0,
                       use_3d_text=False)
        # endregion
        pl.add_mesh(grid.threshold(0.5), color=color, opacity=opacity, edge_color=edge_color, show_edges=show_edges,
                    show_scalar_bar=False, smooth_shading=False)
        # Saving
        if path is not None:
            png = os.path.join(path, name + '.png')
        else:
            path = pathlib.Path.home() / 'Desktop'
            png = os.path.join(path, name + '.png')
        win_size = pl.window_size
        if camera_position is not None:
            pl.camera_position = camera_position
        pl.show(interactive=False, auto_close=True, screenshot=png, window_size=2 * win_size)
    else:
        pl = pv.Plotter(off_screen=False)
        pl.set_background('white')
        #pl.set_background(None)
        pl.remove_bounds_axes()
        #region Bounds
        pl.show_bounds(bounds=[0.0, max(grid.points[:, 0]), 0.0, max(grid.points[:, 1]), 0.0, max(grid.points[:, 2])],
                       font_size=21,
                       font_family='times',
                       bold=False,
                       color='k',
                       xtitle=f'x ({units})',
                       ytitle=f'y ({units})',
                       ztitle=f'z ({units})',
                       grid=False,
                       #location='outer',
                       location='origin',
                       #location='all',
                       #use_2d=True,
                       ticks='inside',
                       show_xlabels=True,
                       show_ylabels=True,
                       show_zlabels=True,
                       n_xlabels=2,
                       n_ylabels=2,
                       n_zlabels=2,
                       minor_ticks=False,
                       fmt='%.1f',
                       padding=0.0,
                       use_3d_text=False)
        #endregion
        pl.add_mesh(grid.threshold(0.5), color=color, opacity=opacity, edge_color=edge_color, show_edges=show_edges,
                    show_scalar_bar=False, smooth_shading=False)
        pl.show(interactive=True, auto_close=True)
    pl.close()
    return grid

def PyVista_M_Voxels(images, voxel_size, threshold=None, units=None, show_edges=False, camera_position=None, legend=False, show_bounds=True, save_fig=False, path=None, name='TPMS'):
    """Intakes a list like [im values, face and edge color, opacity, label] where im values are boolean masks."""

    def np_to_grid(im, voxel_size, threshold=None):
        # Create the spatial reference
        grid = pv.ImageData(origin=(0, 0, 0), spacing=(voxel_size, voxel_size, voxel_size),
                            dimensions=(im.shape[0] + 1, im.shape[1] + 1, im.shape[2] + 1))
        grid.dimensions = np.array(im.shape) + 1
        grid.spacing = (voxel_size, voxel_size, voxel_size)  # These are the cell sizes along each axis
        grid.cell_data["values"] = im.flatten(order='F').astype(np.float32)  # Flatten the array
        grid = grid.threshold() #removes nans
        if threshold is not None:
            grid = grid.threshold(threshold)
        #return grid.extract_geometry()
        return grid

    #region Create plotting window
    pl = pv.Plotter(off_screen=False)
    if save_fig:
        pl = pv.Plotter(off_screen=True)
    pl.set_background('white')
    pl.remove_bounds_axes()
    #endregion

    #region Create and add the meshes
    labels = []
    if not isinstance(images, (np.ndarray, list, tuple)):
        raise ValueError('Provide list of images to add.\n')
    else:
        for i, im in enumerate(images): #im é uma lista
            if not isinstance(im, (np.ndarray, list, tuple)) and len(im) < 3:
                raise ValueError('Provide a list from 3 to 4 elements like: [im values, face and edge color, opacity, label].\n')
            else:
                #im[0] = np.where(im[0], i+1, 0).astype(np.float16) #They are boolean
                im[0] = np.where(im[0], i + 1, 0).astype(np.uint8)  # They are boolean
                #Create grid data for each image
                pl.add_mesh(mesh=np_to_grid(im=im[0], voxel_size=voxel_size, threshold=threshold),
                            color=im[1],
                            opacity=im[2], show_edges=show_edges)
                try:
                    labels.append([im[3], im[1], 'circle'])
                except IndexError:
                    legend = False
                    pass

    #region Legend
    if legend:
        pl.add_legend(labels=labels,
                      bcolor=None,
                      border=False,
                      size=(0.15, 0.15),
                      name=None,
                      loc='upper right',
                      face=None,
                      font_family='times',
                      background_opacity=0.0)
    #endregion

    # region Bounds
    if show_bounds:
        pl.show_bounds(#bounds=[0.0, max(grid.points[:, 0]), 0.0, max(grid.points[:, 1]), 0.0, max(grid.points[:, 2])],
                       font_size=21,
                       font_family='times',
                       bold=False,
                       color='k',
                       xtitle=f'x ({units})',
                       ytitle=f'y ({units})',
                       ztitle=f'z ({units})',
                       grid=False,
                       # location='outer',
                       location='origin',
                       # location='all',
                       # use_2d=True,
                       ticks='inside',
                       show_xlabels=True,
                       show_ylabels=True,
                       show_zlabels=True,
                       n_xlabels=2,
                       n_ylabels=2,
                       n_zlabels=2,
                       minor_ticks=False,
                       fmt='%.1f',
                       padding=0.0,
                       use_3d_text=False)
    # endregion
    #endregion

    #region Saving or Plotting
    if save_fig:
        # Saving
        if path is not None:
            png = os.path.join(path, name + '.png')
        else:
            path = pathlib.Path.home() / 'Desktop'
            png = os.path.join(path, name + '.png')
        win_size = pl.window_size
        if camera_position is not None:
            pl.camera_position = camera_position
        pl.show(interactive=False, auto_close=True, screenshot=png, window_size=2 * win_size)
    else:
        pl.show(interactive=True, auto_close=True)
    #endregion
    cp = pl.camera_position
    pl.close()
    pl.deep_clean()
    return cp