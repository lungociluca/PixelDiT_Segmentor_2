import torch
from tqdm import tqdm
import os

from t2i.diffusion.model.dpm_solver import DPM_Solver
import t2i.config as local_config

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
        t = torch.tensor(local_config.timestep, device=local_config.device, dtype=torch.bfloat16)
        return self.model_fn(x, t) 

    def segment(self, x, segment_data):
        t = torch.tensor(local_config.timestep, device=local_config.device, dtype=torch.bfloat16)
        return self.model(x, t, segment_data=segment_data) 

    # def precompute_language_embs_wrapper(self):
    #     t = torch.tensor(local_config.timestep, device=local_config.device, dtype=torch.bfloat16)
    #     return self.precompute_language_embs(t)