python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_20.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.15_prune_20
	

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_10.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.15_prune_10
	

python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resnet_18_ticket_10.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR output/transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.15_prune_10
	



python tools/train_net.py --num-gpus 4 --resume \
  --config-file  configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.1 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET_TYPE res50 \
	IMAGENET_TICKET  resnet_50_ticket_10.pth  \
	SOLVER.BASE_LR 0.015 \
	OUTPUT_DIR output/transfer_keypoint_rcnn_r50_fpn_warm2k_lr_0.015_batch16_prune_10_late_reset_1112 

	SOLVER.IMS_PER_BATCH 20 \



	LATE_RESET_CKPT  output/keypoint_r50_fpn_1x_1029/model_0003534.pth \
	MODEL.WEIGHTS  output/keypoint_r50_fpn_1x_1029/model_final.pth
	 MODEL.MASK_ON True SOLVER.BASE_LR 0.02 \

python tools/train_net.py --num-gpus 4 --resume  \
 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
 SOLVER.WARMUP_ITERS 2000  \
 SOLVER.WARMUP_FACTOR 5e-4 \
 LOTTERY_KEEP_PERCENTAGE 0.1 \
 NUM_ROUNDS 2 \
 IMAGENET_TICKET resnet_50_ticket_10.pth \
 IMAGENET_TICKET_TYPE res50 \
 OUTPUT_DIR output/transfer_mask_ticket_res50_fpn_warm2k_lr_0.02_prune_10_1111
