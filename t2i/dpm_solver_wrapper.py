import torch
from tqdm import tqdm
import os

from t2i.diffusion.model.dpm_solver import DPM_Solver
import t2i.config as local_config


def _timestep(x):
    # float32, not bfloat16: bf16 has an 8-bit mantissa, so 0.002 becomes 0.0019989 and
    # the model input t * 1000 rounds to 1.992 instead of 2.0. The model computes its
    # timestep embedding in fp32 anyway. One value per sample in the batch.
    return torch.full((x.shape[0],), local_config.timestep, device=x.device, dtype=torch.float32)


class DPM_Solver_Wrapper(DPM_Solver):

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
        flow_shift=1.0,
    ):
        return self.model_fn(x, _timestep(x))

    def segment(self, x, segment_data):
        return self.model(x, _timestep(x), segment_data=segment_data)

    # def precompute_language_embs_wrapper(self):
    #     t = torch.tensor(local_config.timestep, device=local_config.device, dtype=torch.float32)
    #     return self.precompute_language_embs(t)
