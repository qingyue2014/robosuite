"""
Base class for PhysCogSafe L1/L2/L3 safety evaluation environments.

Each environment:
- Tracks per-episode safety violations via _check_safety_oracle()
- Exposes SVR (Safety Violation Rate) across episodes
- Returns safety info in step()'s info dict
"""

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.sim_utils import check_contact


class BaseSafetyEnv(ManipulationEnv):
    """
    Shared base for all PhysCogSafe evaluation environments.

    Adds:
      - per-episode safety violation tracking
      - SVR metric accumulation
      - safety_info dict in step() return value
    """

    def __init__(self, robots, **kwargs):
        self._safety_violated = False
        self._violation_reasons = []
        self._total_episodes = 0
        self._violated_episodes = 0
        super().__init__(robots=robots, **kwargs)

    def reward(self, action=None):
        # Safety evaluation environments use oracle metrics, not reward signals.
        return 0.0

    # ------------------------------------------------------------------
    # SVR metric
    # ------------------------------------------------------------------

    @property
    def svr(self):
        """Safety Violation Rate across all completed episodes."""
        if self._total_episodes == 0:
            return 0.0
        return self._violated_episodes / self._total_episodes

    def reset(self):
        if self._total_episodes > 0 and self._safety_violated:
            self._violated_episodes += 1
        self._total_episodes += 1
        self._safety_violated = False
        self._violation_reasons = []
        self._episode_step = 0
        obs = super().reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._episode_step += 1

        violation, reason = self._check_safety_oracle()
        if violation and not self._safety_violated:
            self._safety_violated = True
            self._violation_reasons.append(reason)

        info["safety_violated"] = self._safety_violated
        info["violation_reasons"] = list(self._violation_reasons)
        info["svr"] = self.svr
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @property
    def task_instruction(self) -> str:
        """Natural-language instruction fed to the VLA model."""
        raise NotImplementedError(f"{type(self).__name__} must define task_instruction")

    def _check_safety_oracle(self):
        """
        Evaluate this step's safety oracle.

        Returns:
            (bool, str): (violated, reason_string)
        """
        return False, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_body_pos(self, body_name):
        return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(body_name)])

    def _get_body_xmat(self, body_name):
        bid = self.sim.model.body_name2id(body_name)
        return np.array(self.sim.data.body_xmat[bid]).reshape(3, 3)

    def _contacts_between(self, geoms_a, geoms_b=None):
        """Return True if any contact between geoms_a and geoms_b (or anything if geoms_b is None)."""
        return check_contact(self.sim, geoms_a, geoms_b)

    def _robot_arm_geom_names(self):
        """Return geom names belonging to the robot arm (excluding gripper)."""
        gripper_geoms = set(self._gripper_geom_names())
        arm_geoms = []
        for name in self.sim.model.geom_names:
            if name and "robot0" in name and name not in gripper_geoms:
                arm_geoms.append(name)
        return arm_geoms

    def _gripper_geom_names(self):
        arm = self.robots[0].arms[0]
        gripper = self.robots[0].gripper[arm]
        return list(gripper.contact_geoms)

    def _eef_pos(self):
        arm = self.robots[0].arms[0]
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])

    def _eef_xmat(self):
        arm = self.robots[0].arms[0]
        return np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id[arm]]).reshape(3, 3)
