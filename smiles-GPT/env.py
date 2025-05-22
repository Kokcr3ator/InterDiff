from typing import Dict, Any
from dataclasses import dataclass, replace
import torch


@dataclass
class Timestep:
    t: int
    """A class to represent a timestep in the environment."""
    obs: torch.Tensor
    """The observation at this timestep."""
    reward: float
    """The reward received at this timestep."""
    terminated: bool
    """Whether the episode has terminated at this timestep."""
    info: Dict[str, Any]
    """Additional information about the timestep."""

    def replace(self, **kwargs):
        """Replace the attributes of the Timestep with new values."""
        return replace(self, **kwargs)


class Env:
    """LLM Environment for Reinforcement Learning"""

    def __init__(self, config: Dict[str, Any]):
        self.discount = config["discount"]
        self.sos_token = config["vocab"]["SOS_TOKEN"]
        self.eos_token = config["vocab"]["EOS_TOKEN"]
        self.pad_token = config["vocab"]["PAD_TOKEN"]
        self.context_length = config["context_length"]

        # init the environment
        self.current_timestep = self.reset()

    def _observation(self) -> torch.Tensor:
        """Get the current observation."""
        return self.current_timestep.obs[: self.current_timestep.t + 1]

    def _reward(self, action: int) -> float:
        """Calculate the reward for the given action."""
        raise NotImplementedError("Reward function not implemented")

    def _termination(self, action: int) -> bool:
        """Terminates when the last token is an EOS token."""
        return action == self.eos_token

    def step(self, action: int) -> Timestep:
        """Select a token and concatenate it to the current sequence."""
        if self.current_timestep.terminated:
            return self.reset()

        # update the observation
        self.current_timestep.obs[self.current_timestep.t + 1] = action

        # update info
        returns = (
            self.current_timestep.info["returns"] + self._reward(action) * self.discount
        )
        self.current_timestep.info["returns"] = returns

        return self.current_timestep.replace(
            obs=self._observation(),
            reward=self._reward(action),
            terminated=self._termination(action),
        )

    def reset(self, seed: int = 0) -> Timestep:
        """Reset the environment to its initial state and return the initial observation."""
        return Timestep(
            t=0,
            obs=torch.ones(self.context_length, dtype=torch.int32) * self.sos_token,
            reward=0.0,
            terminated=False,
            info={"returns": 0.0},
        )
