export DETECTRON2_DATASETS="../datasets"; 
python train_net.py --config-file configs/san_clip_vit_res4_coco.yaml --num-gpus 1 OUTPUT_DIR output