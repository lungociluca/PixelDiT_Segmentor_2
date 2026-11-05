try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import contextlib
import copy
import itertools
import time
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
    SimpleTrainer,
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
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb
import t2i.config as local_config

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast.*is deprecated",
    category=FutureWarning,
)
os.environ["WANDB_MODE"] = "disabled"


def get_grad_accum_steps() -> int:
    steps = int(getattr(local_config, "grad_accumulation", 1) or 1)
    assert steps >= 1, f"grad_accumulation must be >= 1, got {steps}"
    return steps


def overfit_single_batch_enabled() -> bool:
    # Debug mode: train on the first batch over and over. Turn on with OVERFIT_SINGLE_BATCH=1.
    return os.environ.get("OVERFIT_SINGLE_BATCH", "0") == "1"


class GradAccumClipTrainer(SimpleTrainer):
    """
    SimpleTrainer / AMPTrainer replacement with gradient accumulation and clipping.

    One detectron2 "iteration" = one micro-batch of SOLVER.IMS_PER_BATCH images.
    Gradients are summed over `accum_steps` micro-batches (each loss scaled by
    1/accum_steps, so the result is the gradient of the mean loss), then clipped
    once on the full accumulated gradient, then the optimizer steps.
    Effective batch size = IMS_PER_BATCH * accum_steps (* num GPUs).
    """

    def __init__(self, model, data_loader, optimizer, accum_steps=1, clip_cfg=None,
                 amp=False, amp_precision="float16"):
        super().__init__(model, data_loader, optimizer)
        self.accum_steps = accum_steps
        self.clip_cfg = clip_cfg
        self.amp = amp
        self.amp_dtype = getattr(torch, amp_precision) if isinstance(amp_precision, str) else amp_precision
        # A GradScaler is only needed for fp16; bf16 has fp32's exponent range.
        use_scaler = amp and self.amp_dtype == torch.float16
        if not use_scaler:
            self.grad_scaler = None
        elif hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda")
        else:  # older torch
            self.grad_scaler = torch.cuda.amp.GradScaler()
        self._micro_step = 0  # micro-batches accumulated in the current window
        self.max_iter = None  # set by Trainer after construction
        self.overfit_single_batch = overfit_single_batch_enabled()
        self._fixed_batch = None  # cached batch when overfit_single_batch is on

    def _next_batch(self):
        if not self.overfit_single_batch:
            return next(self._data_loader_iter)
        if self._fixed_batch is None:
            self._fixed_batch = next(self._data_loader_iter)
            names = [d.get("file_name", "?") for d in self._fixed_batch]
            logging.getLogger("san").info(
                "[overfit] Training on ONE fixed batch of %d images: %s", len(names), names
            )
        return self._fixed_batch

    def _is_last_micro_step(self) -> bool:
        return (
            self._micro_step + 1 >= self.accum_steps
            or (self.max_iter is not None and self.iter == self.max_iter - 1)  # flush a partial window at the end of training
        )

    def _clip_gradients(self):
        """Clip the accumulated gradients in place; returns the pre-clip total norm."""
        params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
        if not params:
            return None
        cfg = self.clip_cfg
        norm_type = float(cfg.NORM_TYPE) if cfg is not None else 2.0
        enabled = cfg is not None and cfg.ENABLED and cfg.CLIP_VALUE > 0
        clip_type = cfg.CLIP_TYPE if enabled else None

        if clip_type == "full_model":
            # max_norm over all trainable params together
            return torch.nn.utils.clip_grad_norm_(params, cfg.CLIP_VALUE, norm_type=norm_type)

        # measure the total norm for logging without changing anything
        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"), norm_type=norm_type)
        if clip_type == "norm":
            # detectron2 semantics: clip each parameter tensor's norm separately
            for p in params:
                torch.nn.utils.clip_grad_norm_([p], cfg.CLIP_VALUE, norm_type=norm_type)
        elif clip_type == "value":
            torch.nn.utils.clip_grad_value_(params, cfg.CLIP_VALUE)
        elif clip_type is not None:
            raise ValueError(f"Unknown SOLVER.CLIP_GRADIENTS.CLIP_TYPE: {clip_type}")
        return total_norm

    def run_step(self):
        assert self.model.training, "[GradAccumClipTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = self._next_batch()
        data_time = time.perf_counter() - start

        is_step = self._is_last_micro_step()
        # With DDP, skip the gradient all-reduce on micro-batches that don't step.
        sync_ctx = (
            self.model.no_sync()
            if not is_step and isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            else contextlib.nullcontext()
        )
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.amp
            else contextlib.nullcontext()
        )

        with sync_ctx:
            with amp_ctx:
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())
            losses = losses / self.accum_steps
            if self.grad_scaler is not None:
                self.grad_scaler.scale(losses).backward()
            else:
                losses.backward()

        # logs the unscaled per-micro-batch losses, as before
        self._write_metrics(loss_dict, data_time)

        if not is_step:
            self._micro_step += 1
            return

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)  # clip on real gradient values
        grad_norm = self._clip_gradients()
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)  # skips the step on inf/nan grads
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._micro_step = 0

        if comm.is_main_process():
            storage = get_event_storage()
            if grad_norm is not None:
                storage.put_scalar("grad_norm", float(grad_norm), smoothing_hint=False)
            if self.grad_scaler is not None:
                storage.put_scalar("grad_scaler", self.grad_scaler.get_scale(), smoothing_hint=False)

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler is not None:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler is not None and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Replace detectron2's SimpleTrainer / AMPTrainer with one that supports
        # gradient accumulation and clipping. Model (possibly DDP-wrapped), data loader
        # and optimizer are reused, so checkpointing / hooks keep working unchanged.
        inner = self._trainer
        self._trainer = GradAccumClipTrainer(
            inner.model,
            inner.data_loader,
            inner.optimizer,
            accum_steps=get_grad_accum_steps(),
            clip_cfg=cfg.SOLVER.CLIP_GRADIENTS,
            amp=cfg.SOLVER.AMP.ENABLED,
            amp_precision=cfg.SOLVER.AMP.get("PRECISION", "float16"),
        )
        # DefaultTrainer only sets max_iter on itself; the inner trainer needs it to
        # flush a partial accumulation window on the last iteration.
        self._trainer.max_iter = self.max_iter

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
        # ONLY optimize the learnable tokens. Gradient clipping is NOT done here (via an
        # optimizer wrapper) but in GradAccumClipTrainer.run_step, because it has to run
        # once per optimizer step on the accumulated, AMP-unscaled gradients.
        named_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "learnable_token" in name and param.requires_grad
        ]
        assert len(named_params) > 0, "No trainable learnable_token parameter found."

        params = [p for _, p in named_params]
        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                params,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(
                params,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        # display the trainable params and the optimizer settings
        optim_info = defaultdict(list)
        total_params_size = 0
        for name, param in named_params:
            optim_info["Param Name"].append(name)
            optim_info["Param Shape"].append("X".join(str(x) for x in param.shape))
            total_params_size += param.numel()
        optim_info["Param Name"].append("Total")
        optim_info["Param Shape"].append("{:,}".format(total_params_size))
        table = tabulate(
            list(zip(*optim_info.values())),
            headers=optim_info.keys(),
            tablefmt="grid",
            stralign="center",
        )
        clip = cfg.SOLVER.CLIP_GRADIENTS
        logger = logging.getLogger("san")
        logger.info(
            "Optimizer: {} (lr={:.2e}, wd={:.2e}), grad accumulation: {}, grad clipping: {}\n{}\n".format(
                optimizer_type,
                cfg.SOLVER.BASE_LR,
                cfg.SOLVER.WEIGHT_DECAY,
                get_grad_accum_steps(),
                "{} {} (norm_type={})".format(clip.CLIP_TYPE, clip.CLIP_VALUE, clip.NORM_TYPE)
                if clip.ENABLED
                else "off",
                table,
            )
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
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    res = trainer.train()

    # Save the SAN-level learned weights (incl. the logit scale) for later inference runs.
    if local_config.save_learned_tokens and comm.is_main_process():
        model = getattr(trainer.model, "module", trainer.model)  # unwrap DistributedDataParallel
        model.save_learned_weights()
    return res


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
