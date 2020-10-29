#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:gtx1080ti:4
#SBATCH  -c12

cd ~/home/detection/det_oct/ && source activate det_oct

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	OUTPUT_DIR output/mask_r18_fpn_warm1k_b16_lr.015

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.01 \
	OUTPUT_DIR output/mask_r18_fpn_warm1k_b16_lr.01

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.02 \
	OUTPUT_DIR output/mask_r18_fpn_warm1k_b16_lr.02

# python tools/train_net.py --num-gpus 4 \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
# 	MODEL.MASK_ON True \
# 	SOLVER.IMS_PER_BATCH 24 \
# 	SOLVER.BASE_LR 0.01 \
#     SOLVER.WARMUP_ITERS 5000 \
#     SOLVER.WARMUP_FACTOR 2e-5 \
# 	OUTPUT_DIR output/mask_r18_fpn_warm1k_b16_lr.01


# python tools/train_net.py --num-gpus 4 \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
# 	MODEL.MASK_ON True \
# 	SOLVER.BASE_LR 0.015 \
# 	OUTPUT_DIR output/faster_r18_fpn_warm2k_lr.015
