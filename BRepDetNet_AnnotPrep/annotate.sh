rm -rf /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D
mkdir /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D
mkdir /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Scan

folders="171_10011  171_10029  171_10032  171_10045  171_9991" 
for f in $folders; do
    # Generate BRep Boundary to Scan labels 
    python -m BRep2CADLabler labelBRepBndry2Scan --step_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_step/test/batch_01/$f --scan_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_scan/test/batch_01/$f --output /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Bndry --nProc 8 --scale_brep=True --infmt .step --ofmt .npz
    
    # Generate BRep Junction to Scan labels 
    python -m BRep2CADLabler labelBRepJunction2Scan --step_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_step/test/batch_01/$f --scan_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_scan/test/batch_01/$f --output /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Jnc/train --nProc 8 --scale_brep=True --infmt .step --ofmt .npz

    # Generate BRep Face to Scan labels 
    python -m BRep2CADLabler labelBRepFace2Scan --step_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_step/test/batch_01/$f --scan_path /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_scan/test/batch_01/$f --output /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Face/train --nProc 8 --infmt .step --ofmt .npz

    # Copy scan
    cp /home/srikanth/Documents/bits/BRep/datasets/Shared_t2b_v1/t2b_v1.0_scan/test/batch_01/$f/*.ply /home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Scan/.
done

echo "Done..."