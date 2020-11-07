#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/ &&  source activate prune-det

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR output/keypoint_rcnn_r50_fpn_warm5k_lr_0.2_prune_10_late_reset/ \
	LATE_RESET_CKPT  output/keypoint_r50_fpn_1x_1029/model_0003534.pth \
	MODEL.WEIGHTS  output/keypoint_r50_fpn_1x_1029/model_final.pth

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR output/keypoint_rcnn_r50_fpn_warm5k_lr_0.2_prune_10_late_reset/ \
	LATE_RESET_CKPT  output/keypoint_r50_fpn_1x_1029/model_0003534.pth \
	MODEL.WEIGHTS  output/keypoint_r50_fpn_1x_1029/model_final.pth

