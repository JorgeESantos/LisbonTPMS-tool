import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area
from .im_seg_functions import PyVista_TriMeshes_plot
import os, pathlib, stl, math
from itertools import combinations
from collections import Counter

#todo consider delete dependency of numpy-stl and fully use trimesh

#Triangular mesh extraction and basic transformations
def load_stl(file_path):
    """Using numpy-stl, uploads a triangular mesh from a .STL file."""

    # Load the STL file into an STL mesh object
    stl_mesh = stl.mesh.Mesh.from_file(file_path)

    # The points attribute is an (n, 9) array.
    # Reshape it into (n, 3, 3) so that each triangle has 3 vertices of 3 coordinates.
    triangles = stl_mesh.points.reshape(-1, 3, 3)

    # Flatten all triangles to get a list of all vertices (n*3, 3)
    all_vertices = triangles.reshape(-1, 3)

    # Use np.unique to remove duplicates.
    # The return_inverse parameter gives a mapping to the unique vertex indices.
    unique_vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)

    # Reshape the inverse indices into (n, 3) to form the face indices.
    faces = inverse_indices.reshape(-1, 3)

    # return unique_vertices, faces, vnormals
    return unique_vertices, faces

"""def mesh_origin_translation0(mesh=None, vertices=None):
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
        return vertices"""

def mesh_origin_translation(mesh=None, vertices=None):
    if mesh is not None:
        vertices = mesh.vertices  # Extract vertices from the mesh
    elif vertices is None:
        raise ValueError('Please provide an acceptable input.\n')

    # Compute the minimum coordinate along each axis
    min_coords = np.min(vertices, axis=0)

    # Translate vertices so that the minimum coordinate in each axis becomes 0.0
    vertices -= min_coords

    if mesh is not None:
        mesh.vertices = vertices
        return mesh
    return vertices

def mesh_unit_normalization(mesh=None, vertices=None):
    if mesh is not None:
        vertices = mesh.vertices
    elif vertices is None:
        raise ValueError("mesh_unit_normalization: Please provide a mesh or vertices.\n")

    # Translate to origin by subtracting minimum coordinate along each axis
    min_coords = np.min(vertices, axis=0)
    vertices -= min_coords

    # Normalize so that the maximum coordinate along any axis becomes 1.0
    max_coords = np.max(vertices, axis=0)
    vertices /= max_coords

    if mesh is not None:
        mesh.vertices = vertices
        return mesh
    return vertices

"""def mesh_normalization(mesh=None, vertices=None, dimensions=[]):
    #Normalizes the vertices coordinates so that the mesh fits within the specified dimensions
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
        return vertices"""

def mesh_normalization(mesh=None, vertices=None, dimensions=(1.0, 1.0, 1.0)):
    """Normalizes the vertices to fit within the specified bounding dimensions (x, y, z)."""

    #region Ensure dimensions is a 3-tuple
    if isinstance(dimensions, (int, float)):
        dimensions = (dimensions, dimensions, dimensions)
    elif isinstance(dimensions, (list, tuple, np.ndarray)):
        if len(dimensions) != 3:
            raise ValueError('mesh_normalization: "dimensions" must have exactly 3 values (x, y, z).\n')
        dimensions = tuple(dimensions)
    else:
        raise TypeError('mesh_normalization: "dimensions" must be a number or a sequence of 3 numbers.\n')
    #endregion

    # Normalize vertices to unit cube
    if mesh is not None:
        mesh = mesh_unit_normalization(mesh=mesh)
        mesh.vertices *= dimensions  # Element-wise scaling
        return mesh
    elif vertices is not None:
        vertices = mesh_unit_normalization(vertices=vertices)
        vertices *= dimensions
        return vertices
    else:
        raise ValueError('mesh_normalization: Please provide either a mesh or vertices.\n')

#Basic mesh characterization
def is_watertight(faces):
    """Check if a triangular mesh is watertight using itertools.combinations.

    A mesh is watertight if every edge (ignoring order) is shared by exactly two triangles, no more, and no less.

    Parameters:
        triangles (np.ndarray): An (m, 3) array of indices into the vertices array.

    Returns:
        bool: True if the mesh is watertight, False otherwise."""

    # Generator that yields each edge as a sorted tuple
    def generate_edges():
        for face in faces:
            # combinations(face, 2) yields all pairs of vertices in the face
            for edge in combinations(face, 2):
                yield tuple(sorted(edge))

    # Count how many times each edge occurs
    edge_counts = Counter(generate_edges())

    # Every edge must appear exactly twice for a watertight mesh
    return all(count == 2 for count in edge_counts.values())

def mesh_volume(vertices, faces): #Important for geometry volume control
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

"""def get_edges_lenght(mesh=None, vertices=None, faces=None):
    #The edges length distribution is directly related to the mesh resolution.

    if mesh is not None:
        edges = mesh.vertices[mesh.edges_unique]
        edge_lengths = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    elif (vertices is not None) and (faces is not None):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        edges = mesh.vertices[mesh.edges_unique]
        edge_lengths = np.linalg.norm(edges[:, 0, :] - edges[:, 1, :], axis=1)
    return edge_lengths"""

def get_edges_length(mesh=None, vertices=None, faces=None):
    """Computes the lengths of all unique edges in a triangular mesh.

    Parameters:
        vertices (np.ndarray): (n, 3) array of vertex coordinates.
        faces (np.ndarray): (m, 3) array of triangle indices.

    Returns:
        np.ndarray: Array of unique edge lengths."""

    # Check if inputs are valid
    if mesh is None:
        if vertices is None or faces is None:
            raise ValueError("get_edges_length: You must provide either a 'mesh' object or both 'vertices' and 'faces'.\n")

    if mesh is not None:
        vertices = mesh.vertices
        faces = mesh.faces

    # Set to hold unique edges (as sorted tuples)
    edge_set = set()
    for face in faces:
        # Generate all 3 edges of the triangle
        for i, j in combinations(face, 2):
            edge = tuple(sorted((i, j)))
            edge_set.add(edge)

    # Convert to array of vertex index pairs
    unique_edges = np.array(list(edge_set))

    # Compute edge lengths
    edge_vectors = vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    return edge_lengths

"""def mesh_from_array0(im, dimensions, level=None, step_size=1.0, normalized=True):
    
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
            if normalized:
                verts = mesh_normalization(vertices=verts, dimensions=dimensions) #This should only be done to STL and FE generation
            return verts, faces, vnormals
        except Exception as e:
            print(f"An error occurred at mesh_from_array: {e}\n")
    else:
        print("Input image was blank/empty. No mesh generated.\n")"""

#Saving
def STL_from_mesh(mesh=None, vertices=None, faces=None, name='TPMS', path=None):
    """Saves a triangular mesh into a binary .STL file for 3D printing purposes."""

    # Check if inputs are valid
    if mesh is None:
        if vertices is None or faces is None:
            raise ValueError("STL_from_mesh: You must provide either a 'mesh' object or both 'vertices' and 'faces'.\n")

    if mesh is not None:
        vertices = mesh.vertices
        faces = mesh.faces

    #region path creation
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
    file = str(name) + '_' + str(round(max(vertices[:, 0]))) \
           + 'x' + str(round(max(vertices[:, 1]))) \
           + 'x' + str(round(max(vertices[:, 2])))
    path = os.path.join(path, file)
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, 0o777)
    # endregion

    # region STL writting
    STL_path = os.path.join(path, str(name) + '.stl')
    mesh_obj = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    # Set vertices and faces
    mesh_obj.vectors[:] = vertices[faces]
    mesh_obj.save(STL_path)
    del mesh_obj
    # endregion

    #Save mesh data
    with open(os.path.join(path, 'Properties.txt'), 'w') as txt:
        txt.write('Domain boundaries:\n')
        txt.write('x_min {}\n'.format(min(vertices[:, 0])))
        txt.write('x_max {}\n'.format(max(vertices[:, 0])))
        txt.write('y_min {}\n'.format(min(vertices[:, 1])))
        txt.write('y_max {}\n'.format(max(vertices[:, 1])))
        txt.write('z_min {}\n'.format(min(vertices[:, 2])))
        txt.write('z_max {}\n'.format(max(vertices[:, 2])))
        txt.write('\n')
        txt.write('Mesh Properties:\n')
        txt.write('Edge Length mean +- std: ' + f'{np.mean(get_edges_length(vertices=vertices, faces=faces)):.5f} +- {np.std(get_edges_length(vertices=vertices, faces=faces)):.5f}\n')
        txt.write('SA: {}\n'.format(round(mesh_surface_area(verts=vertices, faces=faces), 5)))
        txt.write('Volume: {}\n'.format(round(mesh_volume(vertices, faces), 5)))
        txt.write(f'Mesh SSA: {round(mesh_surface_area(verts=vertices, faces=faces) / mesh_volume(vertices, faces), 5)}\n')
        txt.write('Mesh is watertight: {}\n'.format(str(is_watertight(faces))))
        txt.close()
    print('Files writen.\n')
    return STL_path

def ply_from_mesh(mesh=None, vertices=None, faces=None, name='TPMS', path=None, binary=False):
    """
            Save a triangular mesh as a PLY file (ASCII) with vertex normals.

            Parameters
            ----------
            mesh : object, optional
                Mesh object exposing .vertices, .faces and .vertex_normals
            vertices : (N, 3) array-like, optional
                Vertex coordinates
            faces : (M, 3) array-like, optional
                Triangle indices (0-based)
            vertex_normals : (N, 3) array-like, optional
                Vertex normals
            name : str, default 'TPMS'
                Output file name (without extension)
            path : str, optional
                Directory where the file is saved. If None, uses current working directory.
            """

    if mesh is not None:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        vertex_normals = np.asarray(mesh.vertex_normals)
    else:
        if vertices is None or faces is None:
            raise ValueError("ply_from_mesh: Either 'mesh' or both 'vertices' and 'faces' must be provided.\n")

    # Check1
    if vertex_normals is not None:
        if vertex_normals.shape != vertices.shape:
            raise ValueError("vertices_normals must have the same shape as vertices")

    if path is None:
        path = os.getcwd()

    # Prepare data types (important for binary)
    vertices = vertices.astype(np.float32, copy=False)
    if vertex_normals is not None:
        vertex_normals = vertex_normals.astype(np.float32, copy=False)
    faces = faces.astype(np.int32, copy=False)

    # Prepare path and save directory
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.ply")

    #region Writing the .ply file
    mode = 'wb' if binary else 'w'
    with open(filepath, mode) as f:

        # Helper function to write strings or bytes
        def w(line):
            if binary:
                f.write(line.encode())
            else:
                f.write(line)

        # ------------------ Header ------------------
        if binary:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(b"comment Generated by PLY_from_mesh\n")
            f.write(f"element vertex {vertices.shape[0]}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            if vertex_normals is not None:
                f.write(b"property float nx\n")
                f.write(b"property float ny\n")
                f.write(b"property float nz\n")
            f.write(f"element face {faces.shape[0]}\n".encode())
            f.write(b"property list uchar int vertex_indices\n")
            f.write(b"end_header\n")
        else:
            w("ply\n")
            w("format ascii 1.0\n")
            w("comment Generated by PLY_from_mesh\n")
            w(f"element vertex {vertices.shape[0]}\n")
            w("property float x\n")
            w("property float y\n")
            w("property float z\n")
            if vertex_normals is not None:
                w("property float nx\n")
                w("property float ny\n")
                w("property float nz\n")
            w(f"element face {faces.shape[0]}\n")
            w("property list uchar int vertex_indices\n")
            w("end_header\n")

        # ------------------ Body ------------------
        if not binary:
            # ---------- ASCII ----------
            if vertex_normals is not None:
                vertex_data = np.hstack((vertices, vertex_normals))
                np.savetxt(
                    f,
                    vertex_data,
                    fmt="%.6f %.6f %.6f %.6f %.6f %.6f"
                )
            else:
                np.savetxt(
                    f,
                    vertices,
                    fmt="%.6f %.6f %.6f"
                )

            face_data = np.hstack((
                np.full((faces.shape[0], 1), 3, dtype=np.int32),
                faces
            ))

            np.savetxt(
                f,
                face_data,
                fmt="%d %d %d %d"
            )
        else:
            # ---------- Binary ----------
            if vertex_normals is not None:
                vertex_data = np.hstack((vertices, vertex_normals))
            else:
                vertex_data = vertices

            # Vertices: float32
            vertex_data.astype(np.float32, copy=False).tofile(f)

            # Faces: uchar (3) + int32 indices
            """face_counts = np.full((faces.shape[0], 1), 3, dtype=np.uint8)
            face_indices = faces.astype(np.int32)

            face_data = np.hstack((face_counts, face_indices))
            face_data.tofile(f)"""

            face_dtype = np.dtype([
                ('n', 'u1'),
                ('v0', 'i4'),
                ('v1', 'i4'),
                ('v2', 'i4'),
            ])

            face_data = np.empty(faces.shape[0], dtype=face_dtype)
            face_data['n'] = 3
            face_data['v0'] = faces[:, 0]
            face_data['v1'] = faces[:, 1]
            face_data['v2'] = faces[:, 2]

            face_data.tofile(f)
    #endregion
    print(f'ply_from_mesh: Files writen at "{filepath}"\n')

    return filepath

class mesh_from_array:
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
    def __init__(self, im, level=None, step_size=1.0):
        self.vertices = None
        self.faces = None
        self.vertex_normals = None
        self.level = level

        #if (im.size == 0) or (np.amax(im) == 0): #This might hurt STL_Poro_finder
        if (im.size == 0):
            raise ValueError('mesh_from_array: Provided image is empty.\n')

        if np.all(np.isin(im, [0.0, 1.0])):
            im = im.astype(bool)
        else:
            raise Exception('mesh_from_array: The provided image is not boolean.\n')

        if step_size < 1.0:
            print('step_size cannot be inferior to 1.0.\n')
            self.step_size = 1.0
        else:
            self.step_size = step_size

        # region Extract The mesh
        #Create the black box
        bbox = np.pad(array=im, pad_width=int(math.ceil(step_size)), mode='constant', constant_values=False)
        #bbox = np.pad(array=im, pad_width=int(1), mode='constant', constant_values=False)

        #Mesh extraction
        if np.amax(bbox) > 0.0:
            try:
                vertices, faces, vnormals, _ = marching_cubes((bbox == 0.0), level=self.level, step_size=self.step_size,
                                                              allow_degenerate=False)
                self.vertices = vertices
                self.faces = faces
                self.vertex_normals = vnormals
            except Exception as e: #This should be ok
                print(f"Error in mesh_from_array: {e}")
                self.vertices = None
                self.faces = None
                self.vertex_normals = None
        #endregion

    #region mesh transformations
    def origin_translation(self):
        if self.vertices is not None:
            self.vertices = mesh_origin_translation(vertices=self.vertices)
        else:
            pass

    def apply_scale(self, voxel_spacing):
        if self.vertices is not None:
            self.vertices *= voxel_spacing
            self.voxel_spacing = voxel_spacing
        else:
            pass
    #endregion

    #region mesh characterization
    def is_watertight(self):
        self.is_watertight = is_watertight(self.faces)
        return is_watertight(self.faces) #todo do I want to keep this as a method or a pre-computed attribute?

    def calculate_volume(self):
        if self.vertices is not None:
            #self.volume = mesh_volume(mesh_origin_translation(vertices=self.vertices), self.faces)
            return mesh_volume(mesh_origin_translation(vertices=self.vertices), self.faces)
        else:
            return 0.0

    def calculate_surface_area(self):
        if self.vertices is not None:
            #self.SA = mesh_surface_area(verts=self.vertices, faces=self.faces)
            return mesh_surface_area(verts=self.vertices, faces=self.faces)
        else:
            return 0.0
    #endregion

    def mesh_show(self, color='oldlace', opacity=1.0, show_edges=False):
        PyVista_TriMeshes_plot([[self.vertices, self.faces, color, opacity]], show_edges=show_edges)

