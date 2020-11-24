python tools/train_net.py --num-gpus 4 --resume  \
  --config-file configs/COCO-Detection/faster_rcnn_R_18_C4_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4\
	OUTPUT_DIR temp/
	

# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	LOTTERY_KEEP_PERCENTAGE 0.9 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR temp/


# python tools/train_net.py --num-gpus 1 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR temp/ \
# 	SOURCE_TASK det \
# 	SOURCE_MODEL  output/mask_rcnn_r18_fpn_warm5k_lr_0.15_prune_20_late_reset/model_0007329.pth
# 	#OUTPUT_DIR output/mask_transfer_new_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_40/

# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-InstanceSegmentation/mask_rcnn_r18_FPN_1x.yaml \
# 	MODEL.MASK_ON True \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4\
# 	NUM_ROUNDS 2\
# 	OUTPUT_DIR temp/ \
# 	SOURCE_TASK key \
# 	SOURCE_MODEL output/keypoint_rcnn_r18_fpn_warm2k_lr_0.015_prune_20_late_reset/model_0007069.pth 


# #Kamal
# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_18_ticket_10.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_10


# #SHARATH
# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_18_ticket_20.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_20


# #Hao
# python tools/train_net.py --num-gpus 4 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resnet_18_ticket_50.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR output/mask_transfer_keypoint_ticket_res18_fpn_warm2k_lr_0.015_prune_50	


