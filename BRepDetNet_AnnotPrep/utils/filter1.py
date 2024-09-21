from concurrent.futures import process
import os
import py7zr
from tqdm import tqdm
import shutil
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time

parsenet_ids_path = "/data/3d_cluster/ABC_ParseNet/data/shapes/all_ids.txt"
lines = open(parsenet_ids_path,"r").readlines()
all_ids = [line.rstrip() for line in lines]
parsenet_ids = [parsenet_id.split("_")[0] for parsenet_id in all_ids]
step_paths = [""] * len(all_ids)
obj_paths = [""] * len(all_ids)
stl_paths = [""] * len(all_ids)

def extract_chunk(chunk, chunk_type):
    chunk_str = str(chunk).zfill(4)
    zip = "./%s/abc_%s_%s_v00.7z" % (chunk_type, chunk_str, chunk_type.replace("stl", "stl2"))
    chunk_folder = zip[:-3]
    if os.path.isdir(chunk_folder):
        return zip, chunk_folder
    else:
        os.makedirs(chunk_folder)
        with py7zr.SevenZipFile(zip, mode='r') as z:
            z.extractall(chunk_folder)
    return zip, chunk_folder

if __name__ == "__main__":
    #chunks = np.arange(100)
    chunks = [68]
    # for chunk in tqdm(range(13, num_chunks)):
    for chunk in tqdm(chunks):
        ## Download the chunk.
        os.system("python download_chunks.py step %d" % chunk)
        os.system("python download_chunks.py obj %d" % chunk)
        os.system("python download_chunks.py stl %d" % chunk)

        ## Extracting the chunk.
        print("Extracting chunks..")
        try:
            step_zip, chunk_folder_step = extract_chunk(chunk, "step")
            obj_zip, chunk_folder_obj = extract_chunk(chunk, "obj")
            stl_zip, chunk_folder_stl = extract_chunk(chunk, "stl")
        except:
            ## Pass this chunk.
            continue

        ## Find the matching ids.
        print("Finding the matching ids..")
        model_folders = os.listdir(chunk_folder_step)
        paths_to_delete = model_folders[:]
        for i in range(len(parsenet_ids)):
            model_id = parsenet_ids[i]
            if model_id in model_folders:
                step_path = os.path.join(os.path.abspath(chunk_folder_step), model_id)
                obj_path = os.path.join(os.path.abspath(chunk_folder_obj), model_id)
                stl_path = os.path.join(os.path.abspath(chunk_folder_stl), model_id)
                step_paths[i] = step_path
                obj_paths[i] = obj_path
                stl_paths[i] = stl_path

        for model_id in model_folders:
            if model_id not in parsenet_ids:
                shutil.rmtree(os.path.join(os.path.abspath(chunk_folder_step), model_id))
                shutil.rmtree(os.path.join(os.path.abspath(chunk_folder_obj), model_id))
                shutil.rmtree(os.path.join(os.path.abspath(chunk_folder_stl), model_id))

        ## Delete the zips.
        print("Cleaning the zips..")
        os.remove(step_zip)
        os.remove(obj_zip)
        os.remove(stl_zip)

        ## Write the paths to chunk file.
        parsenet_abc_chunk_path = "./parsenet_abc_%s.txt" % str(chunk).zfill(4)
        parsenet_abc_chunk_file = open(parsenet_abc_chunk_path, "w")
        for i in range(len(all_ids)):
            parsenet_id = all_ids[i]
            parsenet_abc_chunk_file.write("%s %s %s %s\n" % (parsenet_id, step_paths[i], obj_paths[i], stl_paths[i]))
        parsenet_abc_chunk_file.close()

    ## Write the paths.
    parsenet_abc_path = "./parsenet_abc.txt"
    parsenet_abc_file = open(parsenet_abc_path, "w")
    for i in range(len(all_ids)):
        parsenet_id = all_ids[i]
        parsenet_abc_file.write("%s %s %s %s\n" % (parsenet_id, step_paths[i], obj_paths[i], stl_paths[i]))
    parsenet_abc_file.close()
