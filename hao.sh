#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/ &&  source activate prune-det

# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.6 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_60_late_reset_1112  \
# 	LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
# 	MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth


# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.8 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_80_late_reset_1112  \
# 	LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
# 	MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth

# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.9 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_90_late_reset_1114 \
# 	LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
# 	MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	IMAGENET_TICKET  resnet_50_ticket_20.pth  \
	SOLVER.BASE_LR 0.01 \
	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm5k_lr_0.01_batch16_prune_20_late_reset

	SOLVER.IMS_PER_BATCH 24 \


python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.9 \
	NUM_ROUNDS 2 \
	SOLVER.BASE_LR 0.015 \
	SOLVER.IMS_PER_BATCH 12 \
	SOLVER.STEPS  80000,106667 \
	SOLVER.MAX_ITER 120000 \
	OUTPUT_DIR output/keypoint_r50_fpn_warm5k_lr_0.015_batch12_prune_90_late_reset \
	LATE_RESET_CKPT output/keypoint_r50_fpn_1x_1029/model_0003534.pth \
	MODEL.WEIGHTS output/keypoint_r50_fpn_1x_1029/model_final.pth

	SOLVER.BASE_LR 0.01 \
	SOLVER.IMS_PER_BATCH 8 \
	SOLVER.STEPS  120000,160000 \
	SOLVER.MAX_ITER 180000 \


srun --pty --gres=gpu:p6000:4  --qos=high --time=1-12:00:00 -c12  bash

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.9 \
	NUM_ROUNDS 2 \
	SOLVER.BASE_LR 0.02 \
	OUTPUT_DIR output/debug_keypoint_r50_fpn_warm5k_lr_0.02_prune_90_late_reset/ \
	LATE_RESET_CKPT output/keypoint_r50_fpn_1x_1029/model_0003534.pth \
	MODEL.WEIGHTS output/keypoint_r50_fpn_1x_1029/model_final.pth


	OUTPUT_DIR output/keypoint_r50_fpn_warm5k_lr_0.02_prune_90_late_reset/ \

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	IMAGENET_TICKET  resnet_50_ticket_20.pth  \
	SOLVER.BASE_LR 0.015 \
	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm5k_lr_0.015_prune_20_late_reset
	


# SOLVER:
#   STEPS: (120000, 160000)
#   MAX_ITER: 180000
