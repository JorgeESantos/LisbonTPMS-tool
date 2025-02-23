import numpy as np
import pyvista as pv
import os, pathlib, numbers
import scipy.ndimage as spim
import skimage.morphology as skmorph
from porespy.filters import local_thickness, flood_func
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
def trim_floating_artifacts(im, threshold=float(1/8), kernel_size=3):
    """Takes a binary image, identifies and labels the objects pixels by their size.
    Then it deletes objects smaller than min_size (as a proportion of image size)."""

    strel = None
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
        for label in np.unique(regions[regions > 0]):
            #Substitute label for region element size
            regions = np.where(regions == label, np.count_nonzero(np.where(regions == label, True, False)), regions)
    im = np.where(regions < threshold*np.prod(im.shape), False, im)
    return im

def force_binary_fill_holes(im, threshold, kernel_size=3):
    """Intakes a binary image with holes and fills all holes smaller than treshold
    as a proportion of images total elements."""
    im = (spim.binary_fill_holes(im) == 0.0)
    im = trim_floating_artifacts(im=im, threshold=threshold, kernel_size=kernel_size)
    im = im==0.0
    return im

def flood_fill_objects_size(im, kernel_size=3, rel=False):
    """Takes a binary image and labels each object with its respective number of pixels.
    This is the same as porespy.filters.region_size()"""
    if len(im.shape) == 2:
        strel = skmorph.square(kernel_size).astype(bool)
    elif len(im.shape) == 3:
        strel = skmorph.cube(kernel_size).astype(bool)
    regions = spim.label(im, structure=strel)[0] #labels everything non-zero
    if rel:
        return flood_func(im, func=np.count_nonzero, labels=regions) / np.prod(regions.shape)
    else:
        return flood_func(im, func=np.count_nonzero, labels=regions)

def get_index_slice_bounds(object):
    indices = np.argwhere(object)
    bounds = [
        slice(np.amin(indices[:, 0]), np.amax(indices[:, 0])),
        slice(np.amin(indices[:, 1]), np.amax(indices[:, 1])),
        slice(np.amin(indices[:, 2]), np.amax(indices[:, 2]))
    ]
    return bounds

def slice_3d_array(array, bounds):
    return array[bounds[0], bounds[1], bounds[2]]

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
            pl.add_mesh(grid, cmap=cmap, opacity=opacity, clim=[np.amin(im), np.amax(im)],
                        scalar_bar_args=sargs)
        else:
            pl.add_mesh(grid.threshold(threshold), cmap=cmap, opacity=opacity, clim=[np.amin(im), np.amax(im)],
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
            pl.add_mesh(grid, cmap=cmap, opacity=opacity, clim=[np.amin(im), np.amax(im)],
                        scalar_bar_args=sargs)
        else:
            pl.add_mesh(grid.threshold(threshold), cmap=cmap, opacity=opacity, clim=[np.amin(im), np.amax(im)],
                        scalar_bar_args=sargs)
        pl.show(interactive=True, auto_close=True)
    pl.close()
    return grid

def PyVista_TriMeshes_plot(meshes, units='mm', camera_position=None, show_edges=False, save_fig=False, name='TPMS', path=None):
    """Mounts a 3D image with multiple triangular meshes. Each mesh is a list with the following elements:

    [vertices (ndarray), faces (ndarray), face_color (str), color_opacity (float)].
    
    meshes argument should always be a list containing one or more lists like the previous one."""

    if not isinstance(meshes, list):
        raise ValueError('meshes must be a list in which each mesh must be of type list like [vertices, faces, face_color (str), opacity (float)].')

    #region Image
    pl = pv.Plotter(off_screen=False)
    pl.set_background('white')
    pl.remove_bounds_axes()
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
    if len(meshes) >= 1: #If there 8is one or more meshes
        for i, mesh in enumerate(meshes):
            if not isinstance(mesh[3], (numbers.Number, type)): #if opacity is not defined
                mesh[3] = 1.0
            if not isinstance(mesh[2], str): #if it has no color
                print('Uncolored mesh added.\n')
                pl.add_mesh(mesh=pv.PolyData.from_regular_faces(points=mesh[0], faces=mesh[1]), opacity=mesh[3], show_scalar_bar=False)
            else: #if it has color
                print('Colored mesh added.\n')
                pl.add_mesh(mesh=pv.PolyData.from_regular_faces(points=mesh[0], faces=mesh[1]), color=mesh[2], edge_color=mesh[2],
                            opacity=mesh[3], show_edges=show_edges, show_scalar_bar=False)
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
    return

def PyVista_Binary_Voxels(im, voxel_size, units=None, color='oldlace', edge_color='darkgray', show_edges=False, opacity=1.0, camera_position=None, save_fig=False, path=None, name='TPMS'):
    #Building the voxel Image
    grid = pv.ImageData(origin=(0, 0, 0), spacing=(voxel_size, voxel_size, voxel_size), dimensions=(im.shape[0] + 1, im.shape[1] + 1, im.shape[2] + 1))
    grid.cell_data["values"] = im.flatten(order='F').astype(np.uint8)

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
