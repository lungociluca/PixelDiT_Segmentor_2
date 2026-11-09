"""A detectron2 training loop that supports gradient accumulation.

One call to ``run_step`` == one optimizer step. It pulls ``accum_steps``
micro-batches from the loader, back-propagates each one with its loss divided
by ``accum_steps``, and only then clips / steps the optimizer. Because an
"iteration" still means one optimizer step, SOLVER.MAX_ITER, the LR schedule,
EVAL_PERIOD and CHECKPOINT_PERIOD keep their meaning; each iteration simply
sees ``IMS_PER_BATCH * GRAD_ACCUM_STEPS`` images.
"""
import itertools
import time
from contextlib import nullcontext

import torch
from detectron2.engine import SimpleTrainer
from detectron2.utils.events import get_event_storage
from torch.nn.parallel import DistributedDataParallel

class GradAccumTrainer(SimpleTrainer):
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        accum_steps: int = 1,
        amp_dtype: torch.dtype = None,
        clip_type: str = "",
        clip_value: float = 0.0,
        clip_norm_type: float = 2.0,
    ):
        super().__init__(model, data_loader, optimizer)
        assert accum_steps >= 1, f"GRAD_ACCUM_STEPS must be >= 1, got {accum_steps}"
        self.accum_steps = int(accum_steps)
        # amp_dtype: autocast dtype (see san.utils.amp.resolve_amp_dtype), None = fp32.
        self.amp = amp_dtype is not None
        self.precision = amp_dtype if self.amp else torch.float32
        # bf16 has fp32's exponent range, so loss scaling is only needed for fp16.
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp and self.precision == torch.float16
        )
        self.clip_type = clip_type if clip_value > 0 else ""
        self.clip_value = float(clip_value)
        self.clip_norm_type = float(clip_norm_type)

    def _params(self):
        return [
            p
            for p in itertools.chain(*[g["params"] for g in self.optimizer.param_groups])
            if p.grad is not None
        ]

    def run_step(self):
        assert self.model.training, "[GradAccumTrainer] model was changed to eval mode!"
        self.optimizer.zero_grad(set_to_none=True)

        is_ddp = isinstance(self.model, DistributedDataParallel)
        data_time = 0.0
        loss_dict_avg = {}
        for micro_step in range(self.accum_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time += time.perf_counter() - start

            # Under DDP, only all-reduce gradients on the last micro-batch.
            last = micro_step == self.accum_steps - 1
            sync_ctx = self.model.no_sync() if (is_ddp and not last) else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=self.precision, enabled=self.amp):
                    loss_dict = self.model(data)
                    if isinstance(loss_dict, torch.Tensor):
                        loss_dict = {"total_loss": loss_dict}
                    losses = sum(loss_dict.values())
                self.grad_scaler.scale(losses / self.accum_steps).backward()

            for k, v in loss_dict.items():
                v = v.detach() / self.accum_steps
                loss_dict_avg[k] = loss_dict_avg[k] + v if k in loss_dict_avg else v
            # free this micro-batch's graph outputs before loading the next one
            del loss_dict, losses, data

        self.after_backward()

        # Unscale once, on the fully accumulated gradient, so that clipping and the
        # logged norm are in real units (GradScaler.step won't unscale twice).
        self.grad_scaler.unscale_(self.optimizer)
        params = self._params()
        if params:
            if self.clip_type == "full_model":
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, self.clip_value, norm_type=self.clip_norm_type
                )
            else:
                if self.clip_type == "norm":
                    for p in params:
                        torch.nn.utils.clip_grad_norm_(
                            [p], self.clip_value, norm_type=self.clip_norm_type
                        )
                elif self.clip_type == "value":
                    torch.nn.utils.clip_grad_value_(params, self.clip_value)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
            if torch.isfinite(grad_norm):
                # norm of the gradient *before* full-model clipping
                get_event_storage().put_scalar("grad_norm", grad_norm.item(), smoothing_hint=False)

        self._write_metrics(loss_dict_avg, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
