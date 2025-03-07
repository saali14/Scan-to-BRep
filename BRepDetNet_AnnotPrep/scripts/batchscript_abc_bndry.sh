#!/bin/bash
ABC_ROOT_DIR="/netscratch/ali/ABC_filter"  #<--- Change here to your root path
ABC_SCAN_DIR="$ABC_ROOT_DIR/obj"
ABC_STEP_DIR="$ABC_ROOT_DIR/step"
ABC_ANNOT_BNDRY_DIR="$ABC_ROOT_DIR/abc_v1.0_BoundaryLabels"
LOGDIR="$ABC_ROOT_DIR/logs"
LOGFILE="$LOGDIR/logs_jncPrep.txt"

cd ..

index_array=( $(seq 0 99) )  # Merged array of numbers from 0 to 99

for i in ${index_array[@]}
do
    if [ $i -lt 10 ]; then
        s="abc_000${i}_step_v00"
        m="abc_000${i}_obj_v00"
    else
        s="abc_00${i}_step_v00"
        m="abc_00${i}_obj_v00"
    fi
    python3 -m BRep2CADLabler labelBRepBndry2ScanABC --step_path "$ABC_STEP_DIR/$s" --scan_path "$ABC_SCAN_DIR/$m" --output "$ABC_ANNOT_BNDRY_DIR/$s" --nProc 60 --infmt .step --ofmt .npz >> "$LOGFILE" 2>&1
done

