import glob
from pathlib import Path
from os import path, scandir


def list_subfolders(path, include_name=False):
    if include_name:
        return[ (f.path, f.name) for f in scandir(path) if f.is_dir() ]
    else:
        return[ f.path for f in scandir(path) if f.is_dir() ]


def list_files(path, name_filter):
    return [p.name for p in list(Path(path).glob(name_filter))]


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    
 
