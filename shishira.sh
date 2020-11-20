#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/

# mkdir -p /scratch0/shishira/datasets/coco
# /vulcanscratch/shishira/msrsync /fs/vulcan-datasets/coco/images/ /scratch0/shishira/datasets/coco/ -p 40 -P
# /vulcanscratch/shishira/msrsync /fs/vulcan-datasets/coco/annotations /scratch0/shishira/datasets/coco -p 10 -P

# export DETECTRON2_DATASETS=/scratch0/shishira/datasets/


# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET_TYPE res50 \
# 	IMAGENET_TICKET  resnet_50_ticket_10.pth  \
# 	SOLVER.BASE_LR 0.015 \
# 	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res50_fpn_warm2k_lr_0.015_prune_10/


python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	IMAGENET_TICKET  resnet_50_ticket_20.pth  \
	SOLVER.BASE_LR 0.015 \
	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res50_fpn_warm2k_lr_0.015_prune_20/
