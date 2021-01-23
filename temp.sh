
python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
	MODEL.MASK_ON True \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.3 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR temp/ \
	LATE_RESET_CKPT  output/mask_r18_fpn_warm5k_lr.015/model_0007329.pth \
	MODEL.WEIGHTS  output/mask_r18_fpn_warm5k_lr.015/model_final.pth

# python tools/train_net.py --num-gpus 2 --resume \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
# 	MODEL.MASK_ON True \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4\
# 	LOTTERY_KEEP_PERCENTAGE 0.2\
# 	NUM_ROUNDS 2\
# 	IMAGENET_TICKET resnet_18_ticket_20.pth \
# 	IMAGENET_TICKET_TYPE res18\
# 	OUTPUT_DIR temp/


