
cd /cmlscratch/shishira/prune-det/
source det_env/bin/activate

module load cuda
module load cudnn
export CUDA_HOME=/opt/common/cuda/cuda-10.0.130/lib64/

# mkdir -p /scratch0/shishira/datasets/coco
# /cmlscratch/shishira/msrsync datasets/coco/ /scratch0/shishira/datasets/coco/ -p 40 -P
# #/cmlscratch/shishira/msrsync /fs/vulcan-datasets/coco/annotations /scratch0/shishira/datasets/coco -p 10 -P

# export DETECTRON2_DATASETS=/scratch0/shishira/datasets/

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.9 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR output/mask_rcnn_r18_fpn_warm5k_lr_0.015_prune_90_late_reset/ \
	LATE_RESET_CKPT  output/mask_r18_fpn_warm5k_lr.015/model_0007329.pth \
	MODEL.WEIGHTS  output/mask_r18_fpn_warm5k_lr.015/model_final.pth


#output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_50_late_reset/