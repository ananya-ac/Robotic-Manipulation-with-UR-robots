from typing import Dict

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class NullLowdimRunner(BaseLowdimRunner):
    """No in-training rollouts: for tasks evaluated outside this repo (e.g. the ROS/MuJoCo
    stacking sim), where policy rollouts run in a separate process against the live simulator."""

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return dict()
