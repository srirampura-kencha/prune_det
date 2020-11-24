
python tools/train_net.py --num-gpus 4 --resume \
  --config-file configs/COCO-Detection/faster_rcnn_r18_C4_1x.yaml \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.2 \
	NUM_ROUNDS 2 \
	LATE_RESET_CKPT  faster_rcnn_out/faster_rcnn_r18_base/model_0004999.pth \
	OUTPUT_DIR  faster_rcnn_out/faster_rcnn_r18_prune_20/


