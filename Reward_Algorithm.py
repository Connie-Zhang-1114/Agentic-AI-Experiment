import gymnasium as gym
import numpy as np
from collections import defaultdict
from Grid_Maze_360_vision import *

class EmpiricalInstrumentalDivergenceWrapper(gym.Wrapper):
    """
    Intrinsic reward wrapper that implements the Empirical Instrumental
    Divergence reward signal for the control agent.
 
    At each step, computes the directional freedom of the agent's new
    position and adds a scaled quadratic bonus to the environment reward:
 
    intrinsic_reward = (freedom ** 2) * scale
 
    where freedom value is the fraction of the 4 directions
    that aren't blocked by walls or pocket structures. 
 
    The total reward passed to PPO is: env_reward (red ball) + intrinsic_reward.
    """
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale

        self._last_position = None

        self.total_intrinsic_reward = 0.0
        self.step_count = 0
        self.reward_components = defaultdict(list)

    def _get_base_env(self):
        """Traverse the wrapper stack to reach the base CleanTaskEnv."""
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def _get_position(self):
        """Return the agent's current position as a tuple."""
        return tuple(self._get_base_env().agent_pos)

    def _compute_directional_freedom(self, position):
        """
        Compute directional freedom at a given position.
        Counts how many directions are movable,
        checking both exit walls on the current cell and entry walls
        on neighbouring cells. Returns accessible_count / 4.0.
        This method is the same as the one in CleanTaskEnv.
        """
        base_env = self._get_base_env()

        direction_map = {
            (1,  0): ('right', 'left'),
            (-1, 0): ('left',  'right'),
            (0,  1): ('bottom', 'top'),
            (0, -1): ('top',   'bottom'),
        }

        accessible = 0
        for (dx, dy), (exit_dir, entry_dir) in direction_map.items():
            target = (position[0] + dx, position[1] + dy)

            # Boundary Check
            if not (0 < target[0] < base_env.width - 1 and
                    0 < target[1] < base_env.height - 1):
                continue

            # Exit Wall Check
            current_cell = base_env.grid.get(*position)
            if isinstance(current_cell, MultiWallCell):
                if current_cell.has_wall(exit_dir):
                    continue

            # Entry Wall Check
            target_cell = base_env.grid.get(*target)
            if target_cell is not None:
                if target_cell.type == 'wall' and not isinstance(target_cell, MultiWallCell):
                    continue  
                if isinstance(target_cell, MultiWallCell):
                    if target_cell.has_wall(entry_dir):
                        continue  

            accessible += 1

        return accessible / 4.0

    def step(self, action):
        """
        Execute one step, compute the intrinsic reward,
        and return total_reward = env_reward + intrinsic_reward.
        Intrinsic reward is only computed after the first step (when a
        previous position is available to compare to).
        """
        if isinstance(action, np.ndarray):
            action = action.item()

        obs, env_reward, terminated, truncated, info = self.env.step(action)
        new_pos = self._get_position()

        intrinsic_reward = 0.0
        components = {}

        if self._last_position is not None:
            freedom = self._compute_directional_freedom(new_pos)
            freedom_reward = (freedom ** 2) * self.scale
            intrinsic_reward = freedom_reward
            components['freedom'] = freedom_reward

        # Clip intrinsic reward to be positive
        intrinsic_reward = max(0, intrinsic_reward)
        total_reward = env_reward + intrinsic_reward

        # Add reward details in info dict for logging and debugging
        info.update({
            "id_reward": intrinsic_reward,
            "env_reward": env_reward,
            "freedom": self._compute_directional_freedom(new_pos),
            **{f"id_{k}": v for k, v in components.items()}
        })

        self.total_intrinsic_reward += intrinsic_reward
        self.step_count += 1

        for k, v in components.items():
            self.reward_components[k].append(v)

        if self.step_count % 500 == 0:
            self._print_debug(env_reward, intrinsic_reward, total_reward, components)

        self._last_position = new_pos

        return obs, total_reward, terminated, truncated, info

    def _print_debug(self, env_rew, id_rew, total_rew, components):
        """
        Print a periodic summary of reward details.
        Triggered every 500 steps. Shows average intrinsic reward
        and recent average freedom reward over the last 500 steps.
        """
        avg_ir = self.total_intrinsic_reward / self.step_count
        print(f"Step {self.step_count} | Env: {env_rew:.3f} | ID: {id_rew:.3f} | "
              f"Total: {total_rew:.3f} | Avg ID: {avg_ir:.3f}")
        if len(self.reward_components.get('freedom', [])) > 100:
            avg_freedom = np.mean(self.reward_components['freedom'][-500:])
            print(f"  Recent avg freedom reward: {avg_freedom:.3f}")

    def reset(self, **kwargs):
        """Reset the environment and initialize the previous position tracker."""
        obs, info = self.env.reset(**kwargs)
        self._last_position = self._get_position()
        return obs, info

    def get_stats(self):
        """
        Return a summary of intrinsic reward statistics accumulated
        since the last wrapper instantiation. Used by EmpiricalIDCallback.
        """
        stats = {
            "avg_intrinsic_reward": self.total_intrinsic_reward / (self.step_count + 1e-8),
            "total_steps": self.step_count
        }
        if 'freedom' in self.reward_components:
            stats["avg_freedom"] = np.mean(self.reward_components['freedom'])
            stats["std_freedom"] = np.std(self.reward_components['freedom'])
        return stats
