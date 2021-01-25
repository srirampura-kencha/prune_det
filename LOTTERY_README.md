# LTH for object recognition
This repo is the official implementation for the CVPR submission **#10479**

# Installation
For installing the required dependecies to run this repo, please refer to the README.MD document provided by detectron2 and follow the steps. 

# File Structure
The different parts/experiments require switching over to different branches in this repository. The general structure is as follows:
```tools/train_net.py ```:  The main python script to execute for all forms of training and eval. 
```tools/lth.py``` : The LTH class which controls the functions relating to applying LTH on the object recognition models. 
```detectron2/configs/defaults.py```: Contains details about the various parameters that needs to be passed while running ```train_net.py```

# Running experiments
## Direct pruning
```git checkout lottery```: Switch to lottery branch.
Example way to run: 
Here we are training a keypoint detector model with Resnet18 backbone and  with 40% sparsity using direct pruning (all layers are pruned). 
Additional configs are found in ```configs/``` directory.

```
 python tools/train_net.py --num-gpus 4 --resume \
   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
 	SOLVER.BASE_LR 0.015 \
 	SOLVER.WARMUP_ITERS 2000 \
 	 SOLVER.WARMUP_FACTOR 5e-4 \
 	LOTTERY_KEEP_PERCENTAGE 0.6 \
 	NUM_ROUNDS 2 \
 	OUTPUT_DIR temp/
```


## Transfer tickets
```git checkout lottery```: Switch to lottery branch.
Example:
To train a Resnet-18 keypoint detector with pruned imagnet backbone (transferred) at 90% sparsity. 

```
python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_10.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_10
```

## Mask transfer
```git checkout mask_transfer```: Switch to mask_transfer branch
This branch has code to run experiments where only mask (not values) is transferred from imagenet models. 
Example: Training a resnet50 backbone detector where the backbone mask is transferred with 90% sparsity. 
```
python tools/train_net.py --num-gpus 4 --resume \
      --config-file  configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4\
	LOTTERY_KEEP_PERCENTAGE 0.1\
	NUM_ROUNDS 2\
	OUTPUT_DIR output/mask_rcnn_r50_fpn_warm5k_lr_0.2_prune_10_late_reset/ \
	LATE_RESET_CKPT  output/mask_r50_fpn_1x_1029/model_0007329.pth \
	MODEL.WEIGHTS  output/mask_r50_fpn_1x_1029/model_final.pth
```
## Task Transfer
```git checkout task_transfer```: Switch to task transfer branch. 
Here we have code for running experiments relating to transferring masks across tasks. 
Example:
Code to train a Res18 keypoint detector with 80% sparsity, with the mask being obtained from the corresponding detection task. 
```
python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR temp/ \
	SOURCE_TASK det \
	SOURCE_MODEL  output/mask_rcnn_r18_fpn_warm5k_lr_0.15_prune_20_late_reset/model_0007329.pth
	#OUTPUT_DIR output/mask_transfer_new_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_40/
```
