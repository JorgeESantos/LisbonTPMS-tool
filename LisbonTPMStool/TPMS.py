import numpy as np
import numbers
from .Utilities import STL_Poro_finder, im_Poro_finder
from .Surfaces import surfaces_dict
from .im_seg_functions import PyVista_Binary_Voxels, get_largest_object

def TPMS_domain(dimensions, voxel_size, domain_bounds=[-np.pi, np.pi]):
    return np.ogrid[domain_bounds[0]:domain_bounds[1]:complex(round(dimensions[0] / voxel_size)),
                  domain_bounds[0]:domain_bounds[1]:complex(round(dimensions[1] / voxel_size)),
                  domain_bounds[0]:domain_bounds[1]:complex(round(dimensions[2] / voxel_size))]

class TPMS:
    def __init__(self, name=None, grid=None, dimensions=(1.0, 1.0, 1.0), domain_bounds=[-np.pi, np.pi], voxel_size=0.005):
        # region Grid and name entries
        if grid is not None:  # trigger for grid
            self.grid = grid
            self.name = name  # It needs to be defined
        elif name is not None:  # trigger for name
            for i, surface_name in enumerate(surfaces_dict.keys()):
                if name.casefold() in surface_name.casefold():
                    self.name = surface_name
                    break
                else:
                    self.name = None
            if self.name is None:
                print(f'The provided {name} name was not associated with the considered TPMS. Please try again.\n')
            # else:
            # print(f'{self.name} TPMS_domain instance initiated.\n')
        else:
            print('Neither name or grid were provided. Please provide one to start.\n')
            pass
        # endregion

        # region Dimensions and voxel size
        if isinstance(dimensions, (tuple, list, np.ndarray)):
            if len(dimensions) != 3:
                print('dimensions must have 3 values.\n')
                pass
            else:
                self.dimensions = tuple(dimensions)
        elif isinstance(dimensions, (numbers.Number, type)):
            self.dimensions = (dimensions, dimensions, dimensions)
        self.voxel_size = voxel_size
        # endregion

        self.domain = TPMS_domain(dimensions=self.dimensions, voxel_size=self.voxel_size, domain_bounds=domain_bounds)
        print(f'{self.name} instance initiated.\n')

    # region Cell size configuration
    def cell_size_config(self, cell_size=1.0):
        """This method computes the cell size configuration which can be a number or an array,
        with same shape as the domain containing a customized cell_size distribution."""

        f = surfaces_dict[self.name]
        if hasattr(self, 'grid'):  # The cell_config mounts the grid, if the grid has been passed theres no need to do anything
            pass
        else:
            if hasattr(self, 'domain'):
                if isinstance(cell_size, (numbers.Number, type)):  # If it is numeric
                    self.cell_size = cell_size
                    nx, ny, nz = self.dimensions[0] / cell_size, self.dimensions[1] / cell_size, self.dimensions[2] / cell_size
                    # self.cell_number = nx, ny, nz #not sure if this is necessary
                elif isinstance(cell_size, np.ndarray):  # If it is an array
                    self.cell_size = cell_size
                    nx, ny, nz = self.dimensions[0] / cell_size, self.dimensions[1] / cell_size, self.dimensions[2] / cell_size
                    # self.cell_number = nx, ny, nz
                else:
                    print("cell_size must be a number or a np.ndarray.\n")
                    pass
                self.grid = f(nx, ny, nz, self.domain)
            else:
                print('Instance has no attribute domain.\n')
    # endregion

    # region Level-set
    def level_set(self, im_seed=lambda grid, c: np.where((grid >= -c) & (grid <= c), True, False),
                  c=None, target_porosity=None, level=None, step_size=1.0, x0=None, bracket=None, mask=None, mode='STL',
                  trim_artifacts=True, replace=False):

        """This method is deemed to generate the offset based binary image.
        The im_seed defines the type of geometry to generate. Traditional Sheet or Network-based TPMS (defaults to Sheet).
        c is the offset constant that defines the volume of the image and any structure that follows.
        mask is a domain mask, calculated based on the domain (x, y, z) that constrains the geometry (e.g. a cylinder)
        if replace is True, then the original grid (f) and domain variables (x, y, z) will be deleted for memory relief purposes"""

        # Save the level_set condition
        self.im_seed = im_seed

        if (c is None) and (target_porosity is None):
            print('Please provide an offset c or a target porosity value to generate image.\n')
        else:
            if hasattr(self, 'grid'):
                if c is not None:  # c has priority over target_porosity
                    if mask is not None:
                        im = im_seed(self.grid, c)
                        im = mask(im)
                        if trim_artifacts:
                            #im = trim_floating_artifacts(im)
                            im = get_largest_object(im)
                    else:
                        im = im_seed(self.grid, c)
                        if trim_artifacts:
                            #im = trim_floating_artifacts(im)
                            im = get_largest_object(im)
                    self.c = c
                    print('Binary image created.\n')
                elif (c is None) and (target_porosity is not None):  # If c is None, need to provide target_porosity
                    if mode == 'STL':
                        if mask is not None:
                            c = STL_Poro_finder(grid=self.grid, im_seed=self.im_seed, voxel_size=self.voxel_size,
                                                mask=mask, target_porosity=target_porosity,
                                                level=level, step_size=step_size, x0=x0, bracket=bracket)
                            im = im_seed(self.grid, c)
                            im = mask(im)
                            if trim_artifacts:
                                #im = trim_floating_artifacts(im)
                                im = get_largest_object(im)
                        else:
                            c = STL_Poro_finder(grid=self.grid, im_seed=self.im_seed, voxel_size=self.voxel_size,
                                                mask=mask, target_porosity=target_porosity,
                                                level=level, step_size=step_size, x0=x0, bracket=bracket)
                            im = im_seed(self.grid, c)
                            if trim_artifacts:
                                #im = trim_floating_artifacts(im)
                                im = get_largest_object(im)
                    elif mode == 'im':
                        if mask is not None:
                            c = im_Poro_finder(f=self.grid, mask=mask, target_porosity=target_porosity, im_seed=im_seed, x0=x0, bracket=bracket)
                            im = im_seed(self.grid, c)
                            im = mask(im)
                            if trim_artifacts:
                                #im = trim_floating_artifacts(im)
                                im = get_largest_object(im)
                        else:
                            c = im_Poro_finder(f=self.grid, im_seed=im_seed, target_porosity=target_porosity, x0=x0, bracket=bracket)
                            im = im_seed(self.grid, c)
                            if trim_artifacts:
                                #im = trim_floating_artifacts(im)
                                im = get_largest_object(im)
                    self.c = c
                    print('Binary image created.\n')
                self.im = im
            else:
                raise ValueError('Instance has no attribute grid to apply level-set condition.\n')
        if replace:
            del self.grid
            del self.domain
        return
    # endregion

    #region Plot im
    def im_visualize(self, units=None, color='oldlace', edge_color='darkgray', show_edges=False, opacity=1.0, camera_position=None,
                     save_fig=False, name=None, path=None):
        """Nothing too much complicated apart from the camera position.
        The camera allways focus / points out to the geometrical center of the image."""

        if hasattr(self, 'im'):
            if name is None:
                PyVista_Binary_Voxels(im=self.im, voxel_size=self.voxel_size, name=self.name,
                                      units=units, color=color, edge_color=edge_color, show_edges=show_edges, opacity=opacity,
                                camera_position=camera_position, save_fig=save_fig, path=path)
            else:
                PyVista_Binary_Voxels(im=self.im, voxel_size=self.voxel_size, name=name,
                                      units=units, color=color, edge_color=edge_color, show_edges=show_edges,
                                      opacity=opacity,
                                      camera_position=camera_position, save_fig=save_fig, path=path)
        else:
            print('Instance has no attribute im. An image cannot be generated.\n')
            pass
    #endregion