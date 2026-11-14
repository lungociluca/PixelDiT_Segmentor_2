try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import copy
import itertools
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
from san.utils.amp import resolve_amp_dtype
from san.utils.grad_accum_trainer import GradAccumTrainer

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast.*is deprecated",
    category=FutureWarning,
)
os.environ["WANDB_MODE"] = "disabled"

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Swap detectron2's SimpleTrainer/AMPTrainer for one that accumulates
        # gradients and does clipping on the unscaled, accumulated gradient.
        # model / data_loader / optimizer are the same objects DefaultTrainer built
        # (model is already DDP-wrapped when running on several GPUs).
        clip = cfg.SOLVER.CLIP_GRADIENTS
        self._trainer = GradAccumTrainer(
            self._trainer.model,
            self._trainer.data_loader,
            self._trainer.optimizer,
            accum_steps=cfg.SOLVER.GRAD_ACCUM_STEPS,
            amp_dtype=resolve_amp_dtype(cfg),
            clip_type=clip.CLIP_TYPE if clip.ENABLED else "",
            clip_value=clip.CLIP_VALUE,
            clip_norm_type=clip.NORM_TYPE,
        )
        logging.getLogger("san").info(
            "Gradient accumulation: %d micro-batch(es) of %d image(s) -> %d images per optimizer step",
            cfg.SOLVER.GRAD_ACCUM_STEPS,
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.GRAD_ACCUM_STEPS,
        )

    def build_writers(self):
        writers = super().build_writers()
        # use wandb writer instead.
        writers[-1] = WandbWriter()
        return writers

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # resue maskformer dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Add dataset meta info.
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # use poly scheduler
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # All trainable parameters (the frozen PixelDiT has requires_grad=False).
        # Same grouping as upstream SAN: no weight decay on norms, and
        # WEIGHT_DECAY_EMBED on positional embeddings and query embeddings.
        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        embed_names = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "side_adapter_network" in module_name:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                # Concept-classifier scalars (temperature, block weights).
                if module_name.split(".")[-1] == "classifier" and param_name in (
                    "block_logits", "log_temperature"
                ):
                    lr = lr * cfg.MODEL.PIXELDIT.CLASSIFIER.SCALAR_LR_MULTIPLIER
                if any(x in param_name for x in embed_names):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_EMBED
                if isinstance(module, norm_module_types):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_EMBED
                params.append(
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "param_name": ".".join([module_name, param_name]),
                    }
                )

        assert len(params) > 0, "No trainable parameters found."

        # Gradient clipping is applied in GradAccumTrainer.run_step (after the
        # gradients are accumulated and unscaled), not wrapped into the optimizer.
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        logging.getLogger("san").info(
            "Optimizer: AdamW over %d tensors / %.2fM trainable params",
            len(params),
            sum(g["params"][0].numel() for g in params) / 1e6,
        )
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only:
        setup_wandb(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="san")
    return cfg


def write_eval_results(cfg, res):
    """Write eval-only results to OUTPUT_DIR/eval_results.json as one JSON line with the
    same keys as metrics.json ("<dataset>/sem_seg/IoU-<class>", ...), so tools that
    read metrics.json (e.g. tools/class_breakdown.py) work on it too."""
    import json

    # detectron2 returns {dataset: {task: metrics}}, or {task: metrics} for one dataset.
    if not all(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()) for v in res.values()):
        res = {cfg.DATASETS.TEST[0]: res}
    flat = {"iteration": None, "weights": cfg.MODEL.WEIGHTS}
    for dataset, tasks in res.items():
        for task, metrics in tasks.items():
            for k, v in metrics.items():
                flat[f"{dataset}/{task}/{k}"] = float(v)
    path = os.path.join(cfg.OUTPUT_DIR, "eval_results.json")
    with open(path, "a") as f:
        f.write(json.dumps(flat) + "\n")
    logging.getLogger("san").info("Wrote evaluation results to %s", path)


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
            write_eval_results(cfg, res)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
