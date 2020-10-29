python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.02 \
    SOLVER.WARMUP_ITERS 5000 \
    SOLVER.WARMUP_FACTOR 2e-4 \
	OUTPUT_DIR output/mask_r18_fpn_warm5k_b16_lr.02

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.02 \
    SOLVER.WARMUP_ITERS 2500 \
    SOLVER.WARMUP_FACTOR 4e-4 \
	OUTPUT_DIR output/mask_r18_fpn_warm2.5k_b16_lr.02

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
    SOLVER.WARMUP_ITERS 5000 \
    SOLVER.WARMUP_FACTOR 2e-4 \
	OUTPUT_DIR output/mask_r18_fpn_warm5k_b16_lr.015

python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.01 \
    SOLVER.WARMUP_ITERS 5000 \
    SOLVER.WARMUP_FACTOR 2e-4 \
	OUTPUT_DIR output/mask_r18_fpn_warm5k_b16_lr.01


python tools/train_net.py --num-gpus 4 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	OUTPUT_DIR output/faster_r18_fpn_warm2k_lr.015
