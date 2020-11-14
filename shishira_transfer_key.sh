#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH  -c12

# &&  source activate prune-det

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 7000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	IMAGENET_TICKET  resnet_50_ticket_20.pth  \
	SOLVER.BASE_LR 0.008 \
	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm7k_lr_0.008_batch12_prune_20_late_reset/ \
	SOLVER.IMS_PER_BATCH 16 

