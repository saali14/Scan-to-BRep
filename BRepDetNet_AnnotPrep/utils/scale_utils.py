"""
Utility function to scale a solid body into a box
[-1, 1]^3
"""
import copy 
import open3d as o3d
import numpy as np

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_AddOptimal, brepbndlib_Add
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

# occwl
from occwl.solid import Solid

def find_box(solid):
    bbox = Bnd_Box()
    use_triangulation = True
    use_shapetolerance = False
    if solid is None:
        corner1 = gp_Pnt(-1.0, -1.0, -1.0)
        corner2 = gp_Pnt(1.0, 1.0, 1.0)
        bbox = Bnd_Box(corner1, corner2)
        bbox.SetGap(1e-4)
        return bbox 
    else:
        brepbndlib_Add(solid, bbox)
        #brepbndlib_AddOptimal(solid, bbox, use_triangulation, use_shapetolerance)
    return bbox

def scale_solid_to_unit_box(solid):
    is_occwl = False
    if isinstance(solid, Solid):
        is_occwl = True
        topods_solid = solid.topods_solid()
    else:
        topods_solid = solid
    bbox = find_box(topods_solid)
    xmin = 0.0
    xmax = 0.0
    ymin = 0.0
    ymax = 0.0
    zmin = 0.0
    zmax = 0.0
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    longest_length = dx
    if longest_length < dy:
        longest_length = dy
    if longest_length < dz:
        longest_length = dz

    orig = gp_Pnt(0.0, 0.0, 0.0)
    center_array = [
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    ]
    scale = 2.0 / longest_length
    center = gp_Pnt(
        center_array[0],
        center_array[1],
        center_array[2],
    )
    vec_center_to_orig = gp_Vec(center, orig)
    move_to_center = gp_Trsf()
    move_to_center.SetTranslation(vec_center_to_orig)

    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(orig, scale)
    trsf_to_apply = scale_trsf.Multiplied(move_to_center)

    apply_transform = BRepBuilderAPI_Transform(trsf_to_apply)
    apply_transform.Perform(topods_solid)
    transformed_solid = apply_transform.ModifiedShape(topods_solid)

    if is_occwl:
        print("Switch back to occwl solid")
        return Solid(transformed_solid), center_array, scale
    return transformed_solid, center_array, scale

def scale_mesh_to_unit_box(inMesh):
    iMesh = copy.deepcopy(inMesh)
    
    mn = iMesh.get_min_bound()
    mx = iMesh.get_max_bound()
    
    xmin = mn[0]
    xmax = mx[0]
    ymin = mn[1]
    ymax = mx[1]
    zmin = mn[2]
    zmax = mx[2]
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    longest_length = dx
    if longest_length < dy:
        longest_length = dy
    if longest_length < dz:
        longest_length = dz

    orig = [0.0, 0.0, 0.0]
    center = [
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    ]
    
    iMesh.translate(np.asarray(orig) - np.asarray(center))
    iMesh.scale(2.0/longest_length, [0,0,0])
    
    return iMesh

def scale_mesh(inMesh, ctr, s):
    """
    iMesh = copy.deepcopy(inMesh)
    [xmin, ymin, zmin] = iMesh.get_axis_aligned_bounding_box().get_min_bound()
    [xmax, ymax, zmax] = iMesh.get_axis_aligned_bounding_box().get_max_bound()

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    longest_length = dx
    if longest_length < dy:
        longest_length = dy
    if longest_length < dz:
        longest_length = dz

    scaleFactor = 2.0 / longest_length 
    translFactor = np.asarray([-(xmin + xmax) / 2.0, -(ymin + ymax) / 2.0, -(zmin + zmax) / 2.0])
    iMesh.translate(translFactor)
    iMesh.scale(scaleFactor, center=np.array([0,0,0]))
    """

    iMesh = copy.deepcopy(inMesh)
    iMesh.translate(-np.asarray(ctr))
    iMesh.scale(s, [0, 0, 0])
    return iMesh

def scale_Scalar_to_unit(inScalarData):
    return (inScalarData - np.min(inScalarData)) / (np.max(inScalarData) - np.min(inScalarData))
    
