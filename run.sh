export DETECTRON2_DATASETS="../datasets"; 
python train_net.py --config-file configs/san_pixeldit_concept_lora.yaml --num-gpus 1 OUTPUT_DIR output;