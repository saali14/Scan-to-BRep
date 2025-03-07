#!/bin/bash

CC3D_ROOT_DIR="/netscratch/ali/CC3D_filter"   #<--- Change here to your root path
CC3D_SCAN_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_fusion"
CC3D_STEP_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_step"
CC3D_ANNOT_BNDRY_DIR="$CC3D_ROOT_DIR/cc3d_v1.0_BRepBoundaryLabels"
LOGDIR="$CC3D_ROOT_DIR/logs"
LOGFILE="$LOGDIR/logs_bndryPrep.txt"

cd ..

index_array=( $(seq 1 9) )  # Merged array of numbers from 1 to 99
prefixes=("train" "test" "val")  # Define the prefixes for each category

for prefix in "${prefixes[@]}"; do
  for i in "${index_array[@]}"; do
    if [ $i -lt 10 ]; then
      s="${prefix}/batch_0${i}"
      m="${prefix}/batch_0${i}"
    else
      s="${prefix}/batch_${i}"
      m="${prefix}/batch_${i}"
    fi
    python3 -m BRep2CADLabler labelBRepBndry2Scan --step_path "$CC3D_STEP_DIR/$s" --scan_path "$CC3D_SCAN_DIR/$m" --output "$CC3D_ANNOT_BNDRY_DIR/$s" --nProc 60 --infmt .step --ofmt .npz --scale_brep=True >> "$LOGFILE" 2>&1
  done
done

