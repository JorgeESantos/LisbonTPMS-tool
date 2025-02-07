import numpy as np
from skimage.measure import marching_cubes
import os, pathlib, trimesh, pymeshlab, stl, numbers

#Triangular mesh extraction and basic transformations

"""In every function, the mesh argument is a trimesh object resulting from trimesh.Trimesh(vertices, faces)."""

def trimesh_repair_mesh(mesh=None, vertices=None, faces=None, v_normals=None, fix_normals=True, fix_winding=True, fix_inversion=True):
    """Repairs a triangular mesh using trimesh.

    Parameters:
        mesh (trimesh.Trimesh, optional): Input trimesh object.
        vertices (array-like, optional): Vertex data.
        faces (array-like, optional): Face data.
        fix_normals (bool): Whether to fix normals.
        fix_winding (bool): Whether to fix winding consistency.
        fix_inversion (bool): Whether to fix inverted faces.

    Returns:
        trimesh.Trimesh or (numpy.ndarray, numpy.ndarray): Repaired mesh or vertices and faces if input was (vertices, faces)."""

    # Create mesh from vertices and faces if not provided as a trimesh object
    if mesh is None:
        if vertices is None or faces is None:
            raise ValueError("Either a trimesh object or vertices and faces must be provided.")
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=v_normals)
    # Remove infinite values
    mesh.remove_infinite_values()
    # Remove duplicated faces
    mesh.update_faces(mesh.unique_faces())
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    # Fix normals, winding, and inversion
    if not mesh.is_winding_consistent:
        if fix_normals:
            trimesh.repair.fix_normals(mesh, multibody=True)
        if fix_winding:
            trimesh.repair.fix_winding(mesh)
        if fix_inversion:
            trimesh.repair.fix_inversion(mesh, multibody=True)
    # Return appropriate output
    if vertices is not None and faces is not None:
        return mesh.vertices, mesh.faces, mesh.vertex_normals
    return mesh

def mesh_origin_translation(mesh=None, vertices=None):
    if mesh is not None:
        if min(mesh.vertices[:, 0]) > 0.0:
            mesh.vertices[:, 0] = mesh.vertices[:, 0] - abs(min(mesh.vertices[:, 0]))
        if min(mesh.vertices[:, 1]) > 0.0:
            mesh.vertices[:, 1] = mesh.vertices[:, 1] - abs(min(mesh.vertices[:, 1]))
        if min(mesh.vertices[:, 2]) > 0.0:
            mesh.vertices[:, 2] = mesh.vertices[:, 2] - abs(min(mesh.vertices[:, 2]))
        if min(mesh.vertices[:, 0]) < 0.0:
            mesh.vertices[:, 0] = mesh.vertices[:, 0] + abs(min(mesh.vertices[:, 0]))
        if min(mesh.vertices[:, 1]) < 0.0:
            mesh.vertices[:, 1] = mesh.vertices[:, 1] + abs(min(mesh.vertices[:, 1]))
        if min(mesh.vertices[:, 2]) < 0.0:
            mesh.vertices[:, 2] = mesh.vertices[:, 2] + abs(min(mesh.vertices[:, 2]))
        return mesh
    elif vertices is not None:
        if min(vertices[:, 0]) > 0.0:
            vertices[:, 0] = vertices[:, 0] - abs(min(vertices[:, 0]))
        if min(vertices[:, 1]) > 0.0:
            vertices[:, 1] = vertices[:, 1] - abs(min(vertices[:, 1]))
        if min(vertices[:, 2]) > 0.0:
            vertices[:, 2] = vertices[:, 2] - abs(min(vertices[:, 2]))
        if min(vertices[:, 0]) < 0.0:
            vertices[:, 0] = vertices[:, 0] + abs(min(vertices[:, 0]))
        if min(vertices[:, 1]) < 0.0:
            vertices[:, 1] = vertices[:, 1] + abs(min(vertices[:, 1]))
        if min(vertices[:, 2]) < 0.0:
            vertices[:, 2] = vertices[:, 2] + abs(min(vertices[:, 2]))
        return vertices
    return

def mesh_unit_normalization(mesh=None, vertices=None):
    """Normalizes the vertices coordinates so that the mesh boundary dimensions do not exceed 1.0."""
    if mesh != None:
        # region basic translation to 0.0
        if min(mesh.vertices[:, 0]) > 0.0:
            mesh.vertices[:, 0] = mesh.vertices[:, 0] - abs(min(mesh.vertices[:, 0]))
        if min(mesh.vertices[:, 1]) > 0.0:
            mesh.vertices[:, 1] = mesh.vertices[:, 1] - abs(min(mesh.vertices[:, 1]))
        if min(mesh.vertices[:, 2]) > 0.0:
            mesh.vertices[:, 2] = mesh.vertices[:, 2] - abs(min(mesh.vertices[:, 2]))
        if min(mesh.vertices[:, 0]) < 0.0:
            mesh.vertices[:, 0] = mesh.vertices[:, 0] + abs(min(mesh.vertices[:, 0]))
        if min(mesh.vertices[:, 1]) < 0.0:
            mesh.vertices[:, 1] = mesh.vertices[:, 1] + abs(min(mesh.vertices[:, 1]))
        if min(mesh.vertices[:, 2]) < 0.0:
            mesh.vertices[:, 2] = mesh.vertices[:, 2] + abs(min(mesh.vertices[:, 2]))
        # endregion
        #mesh.vertices = mesh.vertices / np.amax(mesh.vertices)
        mesh.vertices[:, 0] = mesh.vertices[:, 0] / np.amax(mesh.vertices[:, 0])
        mesh.vertices[:, 1] = mesh.vertices[:, 1] / np.amax(mesh.vertices[:, 1])
        mesh.vertices[:, 2] = mesh.vertices[:, 2] / np.amax(mesh.vertices[:, 2])
        return mesh
    elif vertices is not None:
        # region basic translation to 0.0
        if min(vertices[:, 0]) > 0.0:
            vertices[:, 0] = vertices[:, 0] - abs(min(vertices[:, 0]))
        if min(vertices[:, 1]) > 0.0:
            vertices[:, 1] = vertices[:, 1] - abs(min(vertices[:, 1]))
        if min(vertices[:, 2]) > 0.0:
            vertices[:, 2] = vertices[:, 2] - abs(min(vertices[:, 2]))
        if min(vertices[:, 0]) < 0.0:
            vertices[:, 0] = vertices[:, 0] + abs(min(vertices[:, 0]))
        if min(vertices[:, 1]) < 0.0:
            vertices[:, 1] = vertices[:, 1] + abs(min(vertices[:, 1]))
        if min(vertices[:, 2]) < 0.0:
            vertices[:, 2] = vertices[:, 2] + abs(min(vertices[:, 2]))
        # endregion
        #vertices = vertices / np.amax(vertices)
        vertices[:, 0] = vertices[:, 0] / np.amax(vertices[:, 0])
        vertices[:, 1] = vertices[:, 1] / np.amax(vertices[:, 1])
        vertices[:, 2] = vertices[:, 2] / np.amax(vertices[:, 2])
        return vertices

def mesh_normalization(mesh=None, vertices=None, dimensions=[]):
    """Normalizes the vertices coordinates so that the mesh fits within the specified dimensions"""
    if isinstance(dimensions, (tuple, list, np.ndarray)):
        if len(dimensions) != 3:
            print('"dimensions" must have 3 values (x, y, z).\n')
        else:
            dimensions = tuple(dimensions)
    elif isinstance(dimensions, (numbers.Number, type)):
        dimensions = (dimensions, dimensions, dimensions)
    if mesh is not None:
        mesh = mesh_unit_normalization(mesh=mesh)
        mesh.vertices[:, 0] = mesh.vertices[:, 0] * dimensions[0]
        mesh.vertices[:, 1] = mesh.vertices[:, 1] * dimensions[1]
        mesh.vertices[:, 2] = mesh.vertices[:, 2] * dimensions[2]
        return mesh
    elif vertices is not None:
        vertices = mesh_unit_normalization(vertices=vertices)
        vertices[:, 0] = vertices[:, 0] * dimensions[0]
        vertices[:, 1] = vertices[:, 1] * dimensions[1]
        vertices[:, 2] = vertices[:, 2] * dimensions[2]
        return vertices

def mesh_from_array(im, dimensions, level=None, step_size=1.0, repair_mesh=False, normalized=True):
    """Generates a triangular mesh from an image represented as a numpy array.

    The level parameter specifies the isovalue of the scalar field in the 3D array at which the isosurface is extracted.
    It determines the "threshold" at which the marching cubes algorithm finds the boundary between two regions
    (e.g., foreground and background in a binary or grayscale image).

    How It Works:

    The algorithm treats the 3D array as a scalar field, where each voxel has an intensity value.
    The level parameter defines the value of this scalar field at which the surface of interest exists.
    For binary data (0 and 1), the typical value of level is 0.5, which corresponds to the midpoint between the two voxel intensities,
    and the fractional distance between the voxel centroids.

    The step_size parameter is basically the marching cubes kernel size. It cannot be inferior to 1.0.
    It specifies how many voxels the algorithm skips when processing the 3D array.

    How It Works:

    The marching cubes algorithm works by evaluating cubes formed by adjacent voxels in the 3D array.
    A step_size of 1 means the algorithm evaluates every voxel and its neighbours,
    resulting in the highest possible resolution for the mesh.
    A larger step_size skips voxels, reducing the number of cubes processed. It should be an integer."""

    if step_size < 1.0:
        step_size = 1.0
        print('step_size cannot be inferior to 1.0.\n')
    bbox = np.zeros([i + 2*round(step_size) for i in im.shape], dtype=bool)
    bbox[round(step_size):-round(step_size), round(step_size):-round(step_size), round(step_size):-round(step_size)] = im

    if np.amax(bbox) > 0.0: #if the image is not blank or empty
        try:
            verts, faces, vnormals, _ = marching_cubes((bbox == 0.0), level=level, step_size=step_size, allow_degenerate=False)
            del bbox
            verts = (verts - 1.0) / (min(im.shape) - 1) * min(dimensions)
            if repair_mesh:
                verts, faces, vnormals = trimesh_repair_mesh(vertices=verts, faces=faces, v_normals=vnormals)
            if normalized:
                verts = mesh_normalization(vertices=verts, dimensions=dimensions) #This should only be done to STL and FE generation
            return verts, faces, vnormals
        except Exception as e:
            print(f"An error occurred at mesh_from_array: {e}\n")
    else:
        print("Input image was blank/empty. No mesh generated.\n")

def trimesh_to_voxel(mesh, voxel_size=1.0):
    """Using trimesh, converts a closed triangular mesh into a voxel representation."""
    voxel_grid = mesh.voxelized(voxel_size).fill()
    matrix = np.array(voxel_grid.matrix).astype(bool)
    return voxel_grid, matrix

#Basic mesh characterization

def Mesh_Volume(vertices, faces): #Important for geometry volume control
    """This function calculates the volume of a closed triangular mesh according to:

    Cha Zhang and Tsuhan Chen,
    "Efficient feature extraction for 2D/3D objects in mesh representation",
    Proceedings 2001 International Conference on Image Processing (Cat. No.01CH37205),
    Thessaloniki, Greece, 2001,
    pp. 935-938 vol.3,
    doi: 10.1109/ICIP.2001.958278."""

    volSTL = 0.0
    for i in range(len(faces)):
        x1 = vertices[faces[i][0], 0]
        y1 = vertices[faces[i][0], 1]
        z1 = vertices[faces[i][0], 2]
        x2 = vertices[faces[i][1], 0]
        y2 = vertices[faces[i][1], 1]
        z2 = vertices[faces[i][1], 2]
        x3 = vertices[faces[i][2], 0]
        y3 = vertices[faces[i][2], 1]
        z3 = vertices[faces[i][2], 2]
        tA = x3 * y2 * z1
        tB = x2 * y3 * z1
        tC = x3 * y1 * z2
        tD = x1 * y3 * z2
        tE = x2 * y1 * z3
        tF = x1 * y2 * z3
        volSTL = 1 / 6 * (-tA + tB + tC - tD - tE + tF) + volSTL
    return volSTL

def get_edges_lenght(mesh=None, vertices=None, faces=None):
    """The edges length distribution is directly related to the mesh resolution."""
    if mesh is not None:
        edges = mesh.vertices[mesh.edges_unique]
        edge_lengths = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    elif (vertices is not None) and (faces is not None):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        edges = mesh.vertices[mesh.edges_unique]
        edge_lengths = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    return edge_lengths

#Advanced Mesh processing

def trimesh_mesh_smoothing(mesh=None, vertices=None, faces=None, vnormals=None,
                           preserve_Boundaries=True, lamb=0.3, iter=60):
    """Laplacian smoothing using trimesh."""

    if mesh is not None:
        if preserve_Boundaries:
            vertices = mesh.vertices
            boundary_vertices = np.argwhere(
                (vertices[:, 0] == min(vertices[:, 0])) | (vertices[:, 1] == min(vertices[:, 1])) | (
                        vertices[:, 2] == min(vertices[:, 2])) |
                (vertices[:, 0] == max(vertices[:, 0])) | (vertices[:, 1] == max(vertices[:, 1])) | (
                        vertices[:, 2] == max(vertices[:, 2])))
            boundary_vertices = np.array([int(i) for i in boundary_vertices])
            lap_operator = trimesh.smoothing.laplacian_calculation(mesh=mesh, equal_weight=True,
                                                                   pinned_vertices=boundary_vertices)
            mesh_smoothed = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iter,
                                                               laplacian_operator=lap_operator, volume_constraint=True)
            mesh_smoothed = trimesh_repair_mesh(mesh_smoothed)
        elif not preserve_Boundaries:
            mesh_smoothed = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iter, volume_constraint=True)
            mesh_smoothed = trimesh_repair_mesh(mesh_smoothed)
        print('Mesh smoothing performed.\n')
        return mesh_smoothed
    elif vertices is not None:
        if vnormals is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vnormals)
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh = trimesh_repair_mesh(mesh)
        if preserve_Boundaries:
            vertices = mesh.vertices
            boundary_vertices = np.argwhere(
                (vertices[:, 0] == min(vertices[:, 0])) | (vertices[:, 1] == min(vertices[:, 1])) | (
                            vertices[:, 2] == min(vertices[:, 2])) |
                (vertices[:, 0] == max(vertices[:, 0])) | (vertices[:, 1] == max(vertices[:, 1])) | (
                            vertices[:, 2] == max(vertices[:, 2])))
            boundary_vertices = np.array([int(i) for i in boundary_vertices])
            lap_operator = trimesh.smoothing.laplacian_calculation(mesh=mesh, equal_weight=True,
                                                                   pinned_vertices=boundary_vertices)
            mesh_smoothed = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iter,
                                                               laplacian_operator=lap_operator, volume_constraint=True)
            mesh_smoothed = trimesh_repair_mesh(mesh_smoothed)
        elif not preserve_Boundaries:
            mesh_smoothed = trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iter, volume_constraint=True)
            mesh_smoothed = trimesh_repair_mesh(mesh_smoothed)
        print('Mesh smoothing performed.\n')
        return mesh.vertices, mesh.faces, mesh.vertex_normals

def pymeshlab_laplcacian_smoothing(iter=2, mesh=None, vertices=None, faces=None, normals=None, preserve_boundary=True):
    """Smoothing using pymeshlab."""
    if mesh is not None:
        m = pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype(np.float64), face_matrix=mesh.faces.astype(np.float64),
                           v_normals_matrix=mesh.vertex_normals.astype(np.float64))
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.set_current_mesh(0)
        if preserve_boundary:
            # Selecting non_boundary vertices
            vertices = ms.current_mesh().vertex_matrix()
            cond_select = f"(x > {min(vertices[:, 0])}) && (x < {max(vertices[:, 0])}) && " \
                          f"(y > {min(vertices[:, 1])}) && (y < {max(vertices[:, 1])}) && " \
                          f"(z > {min(vertices[:, 2])}) && (z < {max(vertices[:, 2])})"
            ms.compute_selection_by_condition_per_vertex(condselect=cond_select)
            # perform the laplacian smoothing
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(iter), boundary=True, cotangentweight=True, selected=True)
            print('Mesh smoothing performed.\n')
        else:
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(iter), boundary=True, cotangentweight=True, selected=False)
            print('Mesh smoothing performed.\n')
        return trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(),
                               faces=ms.current_mesh().face_matrix())
    elif vertices is not None:
        if normals is not None:
            m = pymeshlab.Mesh(vertex_matrix=vertices.astype(np.float64), face_matrix=faces.astype(np.float64),
                               v_normals_matrix=normals.astype(np.float64))
        else:
            m = pymeshlab.Mesh(vertex_matrix=vertices.astype(np.float64), face_matrix=faces.astype(np.float64))
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.set_current_mesh(0)
        if preserve_boundary:
            # Selecting non_boundary vertices
            vertices = ms.current_mesh().vertex_matrix()
            cond_select = f"(x > {min(vertices[:, 0])}) && (x < {max(vertices[:, 0])}) && " \
                          f"(y > {min(vertices[:, 1])}) && (y < {max(vertices[:, 1])}) && " \
                          f"(z > {min(vertices[:, 2])}) && (z < {max(vertices[:, 2])})"
            ms.compute_selection_by_condition_per_vertex(condselect=cond_select)
            # perform the laplacian smoothing
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(iter), boundary=True, cotangentweight=True,
                                               selected=True)
            print('Mesh smoothing performed.\n')
        else:
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=int(iter), boundary=True, cotangentweight=True,
                                               selected=False)
            print('Mesh smoothing performed.\n')
        return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

def pymeshlab_isotropic_remeshing(vertices, faces, vnormals=None, iter=30, targetlen=1.0, adaptative=False, refinestep=True, collapsestep=True, smoothstep=True):
    """Isotropic remeshing using pymeshlab. Mostly used to simplify the mesh although the user can choose decimation if preferred."""
    print('Isotropic remeshing initiated.\n')
    ms = pymeshlab.MeshSet()
    if vnormals is None:
        new_Mesh = pymeshlab.Mesh(vertex_matrix=vertices.astype(np.float64), face_matrix=faces.astype(np.float64))
        ms.add_mesh(new_Mesh)
    else:
        new_Mesh = pymeshlab.Mesh(vertex_matrix=vertices.astype(np.float64), face_matrix=faces.astype(np.float64),
                                  v_normals_matrix=vnormals.astype(np.float64))
        ms.add_mesh(new_Mesh)
    ms.meshing_isotropic_explicit_remeshing(iterations=iter, adaptive=adaptative, selectedonly=False,
                                            targetlen=pymeshlab.PercentageValue(targetlen), featuredeg=30, checksurfdist=True,
                                            maxsurfdist=pymeshlab.PercentageValue(1.0),
                                            splitflag=refinestep, collapseflag=collapsestep, swapflag=True, smoothflag=smoothstep,
                                            reprojectflag=True)
    print('Isotropic remeshing performed.\n')
    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

#Saving

def STL_from_mesh(vertices=None, faces=None, repair=True, name='TPMS', path=None):
    """Saves a triangular mesh into a binary .STL file for 3D printing purposes."""
    if path is not None:
        if not os.path.exists(path):
            os.mkdir(path)
            os.chmod(path, 0o777)
    else:
        dir_f = int(input(
                'Choose directory where to save files:' + '\n' + '1: Desktop' + '\n' + '2: Documents' + '\n' + 'Select: '))
        if dir_f == 1:
            path = pathlib.Path.home() / 'Desktop'
        elif dir_f == 2:
            path = pathlib.Path.home() / 'Documents'
    if (vertices is not None) and (faces is not None):
        file = str(name) + '_' + str(round(max(vertices[:, 0]))) \
               + 'x' + str(round(max(vertices[:, 1]))) \
               + 'x' + str(round(max(vertices[:, 2])))
        path = os.path.join(path, file)
        if not os.path.exists(path):
            os.mkdir(path)
            os.chmod(path, 0o777)
        #Repair the mesh
        if repair:
            vertices, faces, _ = trimesh_repair_mesh(vertices=vertices, faces=faces)
        # region STL writting
        STL_path = os.path.join(path, str(name) + '.stl')
        mesh_obj = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
        # Set vertices and faces
        mesh_obj.vectors[:] = vertices[faces]
        mesh_obj.save(STL_path)
        # endregion
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        del vertices
        del faces
        #Save mesh data
        with open(os.path.join(path, 'Properties.txt'), 'w') as txt:
            txt.write('Domain boundaries:\n')
            txt.write('x_min {}\n'.format(min(mesh.vertices[:, 0])))
            txt.write('x_max {}\n'.format(max(mesh.vertices[:, 0])))
            txt.write('y_min {}\n'.format(min(mesh.vertices[:, 1])))
            txt.write('y_max {}\n'.format(max(mesh.vertices[:, 1])))
            txt.write('z_min {}\n'.format(min(mesh.vertices[:, 2])))
            txt.write('z_max {}\n'.format(max(mesh.vertices[:, 2])))
            txt.write('\n')
            txt.write('Mesh Properties:\n')
            txt.write('Edge Length mean +- std: ' + f'{np.mean(get_edges_lenght(mesh=mesh)):.5f} +- {np.std(get_edges_lenght(mesh=mesh)):.5f}\n')
            txt.write('SA: {}\n'.format(round(mesh.area, 5)))
            txt.write('Volume: {}\n'.format(round(mesh.mass_properties['volume'], 5)))
            txt.write(f'Mesh SSA: {round(mesh.area / mesh.mass_properties["volume"], 5)}\n')
            txt.write('Mesh is watertight: {}\n'.format(str(mesh.is_watertight)))
            txt.close()
        print('Files writen.\n')
    else:
        raise ValueError('Please provide the vertices and faces of the triangular mesh.\n')
    return STL_path

