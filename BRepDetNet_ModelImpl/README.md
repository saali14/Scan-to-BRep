## Command Lines for Training/testing
*	**CC3D** Boundary + Junction Detection 

	Run the following command for **training the Boundary** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 train.py --model_name BRepEd \ 
		--dataset_type CC3D --sampling_type downsample \
		--data_root_dir "$CC3D_ROOT_DIR" --data_scan_dir "$CC3D_SCAN_DIR" \
		--data_annot_bndry_dir "$CC3D_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$CC3D_ANNOT_JNC_DIR" --data_annot_face_dir "$CC3D_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --max_epochs 100 --gpu 1 --num_points 10000 \
		--train_batch_size 6 --eval_batch_size 6 --device "cuda:0" \
		--emb_nn dgcnn --det_loss focal --lr 5e-4
	```

	Run the following command for **training the Junction** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 train.py --model_name BRepJd \
		--dataset_type ABC --sampling_type bjsample \
		--data_root_dir "$ABC_ROOT_DIR" --data_scan_dir "$ABC_SCAN_DIR" \
		--data_annot_bndry_dir "$ABC_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$ABC_ANNOT_JNC_DIR" --data_annot_face_dir "$ABC_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --max_epochs 100 --num_gpus 1 --num_points 4192 \
		--train_batch_size 24 --eval_batch_size 24 --device "cuda:0" \
		--emb_nn dgcnn --det_loss focal --lr 5e-4 --jncOnly 
	```

	Run the following command for **testing the Boundary** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model_name BRepEd \ 
		--dataset_type CC3D --sampling_type downsample \
		--data_root_dir "$CC3D_ROOT_DIR" --data_scan_dir "$CC3D_SCAN_DIR" \
		--data_annot_bndry_dir "$CC3D_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$CC3D_ANNOT_JNC_DIR" --data_annot_face_dir "$CC3D_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --gpu 1 --num_points 10000 \
		--checkpoint_path '/path/to/logs_CC3D/train/BRepEd/version_Id/checkpoint/last.ckpt'
		--train_batch_size 6 --device "cuda:0"
	```

	Run the following command for **testing the Junction** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model_name BRepJd \ 
		--dataset_type CC3D --sampling_type downsample \
		--data_root_dir "$CC3D_ROOT_DIR" --data_scan_dir "$CC3D_SCAN_DIR" \
		--data_annot_bndry_dir "$CC3D_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$CC3D_ANNOT_JNC_DIR" --data_annot_face_dir "$CC3D_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --gpu 1 --num_points 4192 \
		--checkpoint_path '/path/to/logs_CC3D/train/BRepJd/version_Id/checkpoint/last.ckpt'
		--train_batch_size 24 --device "cuda:0"
	```

*	**ABC** Boundary + Junction Detection 

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 train.py --model_name BRepEd \ 
		--dataset_type ABC --sampling_type downsample \
		--data_root_dir "$ABC_ROOT_DIR" --data_scan_dir "$ABC_SCAN_DIR" \
		--data_annot_bndry_dir "$ABC_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$ABC_ANNOT_JNC_DIR" --data_annot_face_dir "$ABC_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --max_epochs 100 --gpu 1 --num_points 10000 \
		--train_batch_size 6 --eval_batch_size 6 --device "cuda:0" \
		--emb_nn dgcnn --det_loss focal --lr 5e-4
	```

	Run the following command for **training the Junction** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 train.py --model_name BRepJd \
		--dataset_type ABC --sampling_type bjsample \
		--data_root_dir "$ABC_ROOT_DIR" --data_scan_dir "$ABC_SCAN_DIR" \
		--data_annot_bndry_dir "$ABC_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$ABC_ANNOT_JNC_DIR" --data_annot_face_dir "$ABC_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --max_epochs 100 --num_gpus 1 --num_points 4192 \
		--train_batch_size 24 --eval_batch_size 24 --device "cuda:0" \
		--emb_nn dgcnn --det_loss focal --lr 5e-4 --jncOnly 
	```

	Run the following command for **testing the Boundary** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model_name BRepEd \ 
		--dataset_type ABC --sampling_type downsample \
		--data_root_dir "$ABC_ROOT_DIR" --data_scan_dir "$ABC_SCAN_DIR" \
		--data_annot_bndry_dir "$ABC_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$ABC_ANNOT_JNC_DIR" --data_annot_face_dir "$ABC_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --gpu 1 --num_points 10000 \
		--checkpoint_path '/path/to/logs_ABC/train/BRepEd/version_Id/checkpoint/last.ckpt'
		--train_batch_size 6 --device "cuda:0"
	```

	Run the following command for **testing the Junction** Detection Network

	```bash
	CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python3 test.py --model_name BRepJd \ 
		--dataset_type ABC --sampling_type downsample \
		--data_root_dir "$ABC_ROOT_DIR" --data_scan_dir "$ABC_SCAN_DIR" \
		--data_annot_bndry_dir "$ABC_ANNOT_BNDRY_DIR" \
		--data_annot_jnc_dir "$ABC_ANNOT_JNC_DIR" --data_annot_face_dir "$ABC_ANNOT_FACE_DIR" \
		--log_dir $LOGDIR --gpu 1 --num_points 4192 \
		--checkpoint_path '/path/to/logs_ABC/train/BRepJd/version_Id/checkpoint/last.ckpt'
		--train_batch_size 24 --device "cuda:0"
	```



Model Checkpoints are available inside [**BRepDetNet_CheckPoints**](../BRepDetNet_CheckPoints) folder 




### CHOSEN UID FOR QUALITATIVE RESULTS

#### TR.ABC+TE.ABC
- 00210432_cc864ae7a3b8df5a0784523f_trimesh_002
- 00216174_0f148b8822a84605928e41d0_trimesh_005
- 00990235_fcce533e2c45f0939d1c82c2_trimesh_000
- 00212258_67dcd284d300f5279b4cdf2c_trimesh_002
- 00214957_57ac73d6e4b005c413ecefe2_trimesh_015 (cylinder)
- 00214172_d3fc0437b02e7d419f5483b5_trimesh_000 (human like)

#### TR.ABC+TE.CC3D
- User Library-Assemblage coque +jante_bouchon reservoir
- User Library-motorcycle bottle opener
- User Library-Parallax Ping Sensor_14 Pin SMT
- User Library-SKKT 132_12 E

## Scan-to-BRep Pretrained Models
Pretrained Models can be found inside **BRepDetNet_CheckPoints** folder
 - Our Model trained on ABC
 - Our Model trained on CC3D

## Authors and acknowledgment
I am thankful to Dr. Anis Kacem and Prof. Dr. Djamila Aouada for sharing appropriate inputs on this project. 
This code has been cleaned up and structured by two undergraduate students - [Bhavika Baburaj](https://github.com/bhavikab04) and [Pritham Kumar Jena](https://github.com/prithamkjena), from BITS Pilani, Hyderabad Campus.

## License
This project is licensed under BSD v3. Please see the LICENSE file. &copy; Copyright @German Research Center for AI (DFKI GmbH) and BITS Pilani, all rights reserved. 

