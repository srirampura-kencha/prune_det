#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH  -c12

cd /fs/vulcan-projects/pruning_sgirish/prune-det/

mkdir -p /scratch0/shishira/datasets/coco
/vulcanscratch/shishira/msrsync /fs/vulcan-datasets/coco/images/ /scratch0/shishira/datasets/coco/ -p 40 -P
/vulcanscratch/shishira/msrsync /fs/vulcan-datasets/coco/annotations /scratch0/shishira/datasets/coco -p 10 -P

export DETECTRON2_DATASETS=/scratch0/shishira/datasets/


# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml  \
# 	MODEL.MASK_ON True \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.1 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_50_ticket_10.pth \
# 	IMAGENET_TICKET_TYPE res50 \
# 	OUTPUT_DIR output/transfer_ticket_res50_fpn_warm5k_lr_0.15_prune_10/

# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml  \
# 	MODEL.MASK_ON True \
# 	SOLVER.BASE_LR 0.010 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.1 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_50_ticket_10.pth \
# 	IMAGENET_TICKET_TYPE res50 \
# 	OUTPUT_DIR output/transfer_ticket_res50_fpn_warm5k_lr_0.10_prune_10/


python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml  \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.008 \
	SOLVER.WARMUP_ITERS 7000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_50_ticket_10.pth \
	IMAGENET_TICKET_TYPE res50 \
	OUTPUT_DIR output/transfer_ticket_res50_fpn_warm7k_lr_0.008_prune_10/
