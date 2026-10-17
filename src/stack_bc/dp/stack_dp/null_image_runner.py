from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NullImageRunner(BaseImageRunner):
    """No in-training rollouts for image policies: rollouts run against the live ROS/MuJoCo sim
    (stack_bc.evaluate + scripts/policy_server.py), as for NullLowdimRunner."""

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return dict()
