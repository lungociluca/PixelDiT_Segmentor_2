from enum import Enum

save_learned_tokens = True 

class EvalDataset(Enum):
    VOC12 = "voc_2012_test_sem_seg"
    ADE20 = "ade20k_sem_seg_{split}".format(split="train" if save_learned_tokens else "val")
    # ADE20 = "ade20k_sem_seg_train"
    COCO = "coco_2017_{split}_stuff_all_sem_seg".format(split="train" if save_learned_tokens else "test")

class SegmentDataKeys(Enum):
    MASK = "mask"
    

ds_config_dict = {
    EvalDataset.VOC12: "t2i/catseg_configs/voc20.json",
    EvalDataset.ADE20: "t2i/catseg_configs/ade150.json"
}

dataset_paths = {
    EvalDataset.ADE20: {
        "json": "t2i/catseg_configs/ade150.json",
        "img_dir": "images",
        "gt_dir": "annotations",
        "extention": ".png",
        "label_count": 150
    },
    EvalDataset.VOC12: {
        "json": "t2i/catseg_configs/voc20.json",
        "img_dir": "JPEGImages",
        "gt_dir": "SegmentationClassAug",
        "extention": ".png",
        "label_count": 20
    },
    EvalDataset.COCO: {
        "json": "t2i/catseg_configs/coco.json",
        "img_dir": "images",
        "gt_dir": "annotations",
        "extention": ".png",
        "label_count": 181,
    }
}

tmp = """
small or large [TARGET]; near or distant [TARGET]; visible, partially visible, occluded, or truncated [TARGET]; isolated, overlapping, or crowded [TARGET]; [TARGET] defined by its overall shape; [TARGET] defined by its distinctive parts; [TARGET] defined by its visual appearance, texture, or material; [TARGET] defined by its local boundaries and separation from surrounding regions.
"""

default_prompt = 'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt: If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes. If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating. Here are examples of how to transform or refine prompts: User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers. User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers. Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations: User Prompt: A picture with a complex scene, multiple objects, maybe sky, vegetation.'
subprompt = 'A picture of a {target}'
prompt_format = 'A picture of a {target}'

device = "cuda"
eval_dataset = EvalDataset.COCO
current_ds_paths = dataset_paths[eval_dataset]
ds_config_path = current_ds_paths["json"]
label_count = current_ds_paths["label_count"]
image_size = 512

epochs = 1
lr = 1e-4
idx_token_of_interest = 0
eval_samples_limit = 3
target_layer = 0
layer_count = 3
timestep = 0.002
extra_labels_count = 2
select_labels_count = 5

grad_accumulation = 8

no_leanable_tokens = 1
no_vision_learnable_tokens = 50
l2_regularization_weight = 0.
diversity_regularization_weight = 0.
crop_h = 448
crop_w = 448

shuffle = False # TODO
run_on_extra_labels = False
use_attention_bias = False
use_vision_attention_bias = False
use_vision_mlp = True
use_text_mlp = False
norm_before_dot = True

crop_size = None
compute_model_vectors = False
use_model_vectors = False

# TODO: update
use_learned_tokens = not save_learned_tokens

# Class names used for the text embeddings of each detectron2 dataset (by registered name).
# Datasets not listed here use MetadataCatalog.get(name).stuff_classes.
# The order must match the contiguous class ids in that dataset's GT masks.
label_json_overrides = {
    "coco_2017_train_stuff_sem_seg": dataset_paths[EvalDataset.COCO]["json"],
    "coco_2017_test_stuff_sem_seg": dataset_paths[EvalDataset.COCO]["json"],
}