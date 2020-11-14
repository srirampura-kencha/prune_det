source det_env/bin/activate

module load cuda
module load cudnn
export CUDA_HOME=/opt/common/cuda/cuda-10.0.130/lib64/


python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.5 \
	NUM_ROUNDS 2 \
    SOLVER.IMS_PER_BATCH 12 \
	SOLVER.BASE_LR 0.012 \
	OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.012_batch12_prune_50_late_reset_1112 \
	LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
	MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.9 \
	NUM_ROUNDS 2 \
    SOLVER.IMS_PER_BATCH 12 \
	SOLVER.BASE_LR 0.012 \
	OUTPUT_DIR output/keypoint_rcnn_r18_fpn_warm2k_lr_0.012_batch12_prune_90_late_reset_1112 \
	LATE_RESET_CKPT  output/keypoint_r18_fpn_warm1k_lr.015/model_0003534.pth \
	MODEL.WEIGHTS  output/keypoint_r18_fpn_warm1k_lr.015/model_final.pth
