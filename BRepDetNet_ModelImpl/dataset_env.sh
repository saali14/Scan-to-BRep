#!/bin/bash
#Environment variables for CC3D and ABC Datasets

# CC3D directories
export CC3D_ROOT_DIR="/path/to/CC3D"
export CC3D_SCAN_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_scans"
export CC3D_ANNOT_BNDRY_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_BoundaryLabels"
export CC3D_ANNOT_JNC_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_JunctionLabels"
export CC3D_ANNOT_FACE_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_BRepFaceLabels"

# ABC directories
export ABC_ROOT_DIR="/path/to/ABC"
export ABC_SCAN_DIR="$ABC_ROOT_DIR/obj"
export ABC_ANNOT_BNDRY_DIR="$ABC_ROOT_DIR/abc_v1.0_BoundaryLabels"
export ABC_ANNOT_JNC_DIR="$ABC_ANNOT_BNDRY_DIR/abc_v1.0_JunctionLabels"
export ABC_ANNOT_FACE_DIR="$ABC_ANNOT_BNDRY_DIR/abc_v1.0_BRepFaceLabels"
export LOGDIR="/mnt/isilon/username/logs_ABC"

