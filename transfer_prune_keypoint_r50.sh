
#NAN
# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
# 	SOLVER.WARMUP_ITERS 4000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.1 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET_TYPE res50 \
# 	IMAGENET_TICKET  resnet_50_ticket_10.pth  \
# 	SOLVER.BASE_LR 0.010 \
# 	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm4k_lr_0.010_batch12_prune_10_late_reset/ \
# 	SOLVER.IMS_PER_BATCH 12 \

python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	IMAGENET_TICKET  resnet_50_ticket_10.pth  \
	SOLVER.BASE_LR 0.008 \
	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm5k_lr_0.008_batch12_prune_10_late_reset/ \
	SOLVER.IMS_PER_BATCH 12






# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.1 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_18_ticket_10.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.15_prune_10
# 	