import os, pathlib, re
from collections import defaultdict
import numpy as np

def sanitize_part_name(part_name, max_length=32):
    """Sanitize a string to be used as a part name in a Finite Element model
    by removing problematic characters.

    Parameters:
        part_name (str): The original name of the part.
        max_length (int): The maximum allowable length for the part name. Default is 32.

    Returns:
        str: A sanitized version of the part name with problematic characters removed."""

    # Remove any disallowed characters by keeping only alphanumeric and underscores
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '', part_name)

    # Ensure the name starts with a letter if it doesn't already
    if not sanitized_name[0].isalpha():
        sanitized_name = 'A' + sanitized_name  # Prefix with 'A' if it starts with a non-letter

    # Truncate to max length if necessary
    if len(sanitized_name) > max_length:
        sanitized_name = sanitized_name[:max_length]

    return sanitized_name

def im_to_C3D8_hex_mesh(im, voxel_size, name='TPMS', path=None):
    """Converts a labelled image to a hexahedral 8-node mesh.
    Saves it in ABAQUS .inp file format."""
    im = im.astype(np.uint8)
    #region Create the path to save data
    if path is not None:
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
            os.chmod(path, 0o777)
    else:
        #Saves it to Desktop
        path = os.path.join(pathlib.Path.home() / 'Desktop', f'{name} inp files')
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
            os.chmod(path, 0o777)
    #endregion
    #Check if the image is empty
    if np.amax(im) == 0:
        print('Image is blank.\n')
    else:
        Elsets_dict = defaultdict(list)
        nodes_ind = []
        elems = []
        elem_ind = 1
        #region Writing Elements
        for label in np.unique(im[im > 0]): #disregards 0s
            print(f'Processing Elset-{label}')
            # Find indices of pixels with value `label`
            #indices = np.argwhere(im == label)
            # Calculate the base index for each pixel
            base_index = np.argwhere(im == label)[:, 2] + 1 + np.argwhere(im == label)[:, 1] * (im.shape[2] + 1) + np.argwhere(im == label)[:, 0] * ((im.shape[2] + 1) * (im.shape[1] + 1))
            # Calculate the other 7 indices relative to the base index
            offsets = np.array([[0, ((im.shape[2] + 1) * (im.shape[1] + 1)), ((im.shape[2] + 1) * (im.shape[1] + 1)) + (im.shape[2] + 1),
                                 (im.shape[2] + 1), 1, ((im.shape[2] + 1) * (im.shape[1] + 1)) + 1, ((im.shape[2] + 1) * (im.shape[1] + 1)) + (im.shape[2] + 1) +1,
                                 (im.shape[2] + 1) + 1]])
            # Create the `elem` array for each pixel
            elem = base_index[:, None] + offsets
            # Create a 2D array with sequential element indices
            element_indices = np.arange(elem_ind, elem_ind + len(elem)).reshape(-1, 1)
            # Concatenate the element indices with the `elem` array
            elems.append(np.hstack((element_indices, elem)))
            # Update `Elsets_dict` with the current range of indices
            Elsets_dict[str(label)].extend(range(elem_ind, elem_ind + len(elem)))
            # Increment `elem_ind` for the next set of elements
            elem_ind += len(elem)
            # Append to `nodes_ind` and `elems`
            nodes_ind.extend(elem.flatten())
            print(f'Elset-{label} processed.\n')
        # Convert `elems` to a NumPy array
        elems = np.vstack(elems)
        print('Element data processed.\n')
        # endregion
        # region Writting Nodes
        print('Writing nodes.')
        # Create a 3D meshgrid of indices
        indices = np.mgrid[0:im.shape[0] + 1, 0:im.shape[1] + 1, 0:im.shape[2] + 1]
        # Flatten the indices and calculate node numbers
        indices = indices.reshape(3, -1).T
        node_numbers = indices[:, 0] * (im.shape[2] + 1) * (im.shape[1] + 1) + indices[:, 1] * (
                    im.shape[2] + 1) + indices[:, 2] + 1
        # Filter for nodes in `nodes_ind`
        mask = np.isin(node_numbers, np.unique(np.array(nodes_ind)))
        #mask = np.isin(node_numbers, np.unique(elems.flatten))
        filtered_indices = indices[mask]
        filtered_node_numbers = node_numbers[mask]
        # Calculate node coordinates
        node_coords = filtered_indices * voxel_size
        # Create the `nodes` array
        nodes = np.hstack((filtered_node_numbers[:, None], node_coords))
        print('Nodes writen.\n')
        # endregion
        #Create the .inp and txt folder
        folder = os.path.join(path, name)
        if os.path.exists(folder):
            pass
        else:
            os.mkdir(folder)
            os.chmod(folder, 0o777)
        # region Writing .inp
        print('Generating .inp files.')
        # Open the file
        inp_file = open(os.path.join(folder, f'{name}.inp'), 'w')
        # Header
        print(f'*Part, name={sanitize_part_name(name)}', file=inp_file)
        # Write the nodes
        print('*Node', file=inp_file)
        np.savetxt(inp_file, nodes, fmt='%d, %.5f, %.5f, %.5f')
        # Write Elems
        print('*Element, type=C3D8', file=inp_file)
        # Flatten the 3D array into a 2D array
        flattened_elems = elems.reshape(-1, 9)
        # Save the flattened array to a text file
        np.savetxt(inp_file, flattened_elems, fmt='%d, ')
        # Write Sets
        for set in Elsets_dict.keys():
            print(f'*Elset, elset=set-{set}', file=inp_file)
            np.savetxt(inp_file, Elsets_dict[set], fmt='%d,')
        print('*End Part', file=inp_file)
        inp_file.close()
        print('Files written.\n')
        #endregion
        #region Writting data txt
        with open(os.path.join(folder, f'{name}_data.txt'), 'w') as txt:
            txt.write(f'Image shape: {im.shape}\n')
            txt.write(f'Voxel size: {voxel_size}\n')
            txt.write(f'Non-zero elements: {np.count_nonzero(im)}\n')
            txt.write(f'Number of nodes: {len(nodes_ind)}\n')
            txt.close()
        #endregion
        return nodes, elems