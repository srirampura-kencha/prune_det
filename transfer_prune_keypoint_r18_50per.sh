#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/ 
# &&  source activate prune-det


# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.2 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_18_ticket_50.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.15_prune_50
	
python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.010 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_50.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm5k_lr_0.010_prune_50 \
	SOLVER.IMS_PER_BATCH 20 \

