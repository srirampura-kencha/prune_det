#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/ &&  source activate prune-det

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_20.pth \
	IMAGENET_TICKET_TYPE res18\
	OUTPUT_DIR output/transfer_keypoint ticket_res18_fpn_warm2k_lr_0.15_prune_20
	

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_10.pth \
	IMAGENET_TICKET_TYPE res18\
	OUTPUT_DIR output/transfer_keypoint ticket_res18_fpn_warm2k_lr_0.15_prune_10
	

	# OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_20_late_reset/ \
	# LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
	# MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth

