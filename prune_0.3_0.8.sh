module load cuda
module load cudnn
export CUDA_HOME=/opt/common/cuda/cuda-10.0.130/lib64/


cd /fs/vulcan-projects/pruning_sgirish/prune-det/

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.3 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR output/mask_rcnn_r18_fpn_warm5k_lr_0.015_prune_30_late_reset/ \
	LATE_RESET_CKPT  output/mask_r18_fpn_warm5k_lr.015/model_0007329.pth \
	MODEL.WEIGHTS  output/mask_r18_fpn_warm5k_lr.015/model_final.pth

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.8 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR output/mask_rcnn_r18_fpn_warm5k_lr_0.015_prune_80_late_reset/ \
	LATE_RESET_CKPT  output/mask_r18_fpn_warm5k_lr.015/model_0007329.pth \
	MODEL.WEIGHTS  output/mask_r18_fpn_warm5k_lr.015/model_final.pth
