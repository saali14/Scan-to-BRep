import numpy as np
# from plyfile import plyData
import os
from glob import glob
import open3d as o3d
import pynanoflann


def root_dict(path_name):
    # Initialize an empty dictionary to store file roots
    roots = {}

    # Get a list of all files in the specified path
    all_files = glob(os.path.join(path_name, "*"))

    # Iterate through each file in the list
    for file in all_files:
        # Extract the root of the file (excluding the file extension)
        root_file = file.split(".")[0]

        # Try to append the current file to the list of files for the corresponding root
        try:
            roots[root_file].append(file)
        # If the root does not exist in the dictionary, create a new entry with the current file
        except KeyError:
            roots[root_file] = [file]

    # Return the dictionary containing file roots as keys and corresponding files as values
    return roots

def read_ply_file(file_path):
    data=o3d.io.read_point_cloud(file_path)
    return data

def read_ply_xyz(path_list):
    """
    Reads .ply files from the provided list of file paths and categorizes them based on file type.
    
    Parameters:
    - path_list (list): List of file paths to be processed.

    Returns:
    A dictionary containing categorized data:
    - 'bndry': Boundary .ply file data (if present).
    - 'jnc': Junction .ply file data (if present).
    - 'input': Input .ply file data (if present).
    """
    data={}
    for path in path_list[2]:
        if path.split(".")[-1]=="ply":
            data['bndry']=np.asarray(read_ply_file(path).points)
        elif "corner" in path.split(".")[1]:
            data['jnc']=np.asarray(read_ply_file(path).points)
        else:
             data['scan']=np.asarray(read_ply_file(path).points)

    data['input']=np.asarray(read_ply_file(path_list[0]).points)
    data['jnc_input']=np.asarray(read_ply_file(path_list[1]).points)

    return data

def voxel2scan_label_mapper(path_list):
    """
    Maps the boundary and junction labels from voxels to scans
    """
    pc_data=read_ply_xyz(path_list)
    label={"bndry":np.zeros(pc_data['input'].shape[0]),
       "jnc":np.zeros(pc_data['jnc_input'].shape[0])}
    
    nn = pynanoflann.KDTree(n_neighbors=1, metric='L1', radius=100)
    scale=np.max(np.stack([np.asarray(pc_data['scan']).max(axis=0),
                           np.abs(np.asarray(pc_data['scan']).min(axis=0))]),axis=0)
    
    divider=np.max(np.stack([np.asarray(pc_data['input']).max(axis=0),
                            np.abs(np.asarray(pc_data['input']).min(axis=0))]),axis=0)
    
    nn.fit(pc_data['input']*scale/divider)

    label['bndry'][np.unique(nn.kneighbors(pc_data['bndry'])[1].reshape(-1))]=1

    # Junction Divider
    nn = pynanoflann.KDTree(n_neighbors=1, metric='L1', radius=100)
    divider_jnc=np.max(np.stack([np.asarray(pc_data['jnc_input']).max(axis=0),
                            np.abs(np.asarray(pc_data['jnc_input']).min(axis=0))]),axis=0)
    
    nn.fit(pc_data['jnc_input']*scale/divider_jnc)
    label['jnc'][np.unique(nn.kneighbors(pc_data['jnc'])[1].reshape(-1))]=1

    return label
