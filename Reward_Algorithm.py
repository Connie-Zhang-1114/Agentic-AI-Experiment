import gymnasium as gym
import numpy as np
from collections import defaultdict
from Grid_Maze import *

class EmpiricalInstrumentalDivergenceWrapper(gym.Wrapper):
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale

        self._last_position = None

        self.total_intrinsic_reward = 0.0
        self.step_count = 0
        self.reward_components = defaultdict(list)

    def _get_base_env(self):
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def _get_position(self):
        return tuple(self._get_base_env().agent_pos)

    def _compute_directional_freedom(self, position):
        """
        计算位置的方向自由度 (0-1)
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

            # 边界检查
            if not (0 < target[0] < base_env.width - 1 and
                    0 < target[1] < base_env.height - 1):
                continue

            # 检查当前格子的出口方向是否有墙
            current_cell = base_env.grid.get(*position)
            if isinstance(current_cell, MultiWallCell):
                if current_cell.has_wall(exit_dir):
                    continue

            # 检查目标格子的入口方向是否有墙
            target_cell = base_env.grid.get(*target)
            if target_cell is not None:
                if target_cell.type == 'wall' and not isinstance(target_cell, MultiWallCell):
                    continue  # 实心墙
                if isinstance(target_cell, MultiWallCell):
                    if target_cell.has_wall(entry_dir):
                        continue  # 单面墙阻挡入口

            accessible += 1

        return accessible / 4.0

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        obs, env_reward, terminated, truncated, info = self.env.step(action)
        new_pos = self._get_position()

        intrinsic_reward = 0.0
        components = {}

        if self._last_position is not None:
            freedom = self._compute_directional_freedom(new_pos)
            freedom_reward = freedom * self.scale
            intrinsic_reward = freedom_reward
            components['freedom'] = freedom_reward

        intrinsic_reward = max(0, intrinsic_reward)
        total_reward = env_reward + intrinsic_reward

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
        avg_ir = self.total_intrinsic_reward / self.step_count
        print(f"Step {self.step_count} | Env: {env_rew:.3f} | ID: {id_rew:.3f} | "
              f"Total: {total_rew:.3f} | Avg ID: {avg_ir:.3f}")
        if len(self.reward_components.get('freedom', [])) > 100:
            avg_freedom = np.mean(self.reward_components['freedom'][-500:])
            print(f"  Recent avg freedom reward: {avg_freedom:.3f}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_position = self._get_position()
        return obs, info

    def get_stats(self):
        stats = {
            "avg_intrinsic_reward": self.total_intrinsic_reward / (self.step_count + 1e-8),
            "total_steps": self.step_count
        }
        if 'freedom' in self.reward_components:
            stats["avg_freedom"] = np.mean(self.reward_components['freedom'])
            stats["std_freedom"] = np.std(self.reward_components['freedom'])
        return stats
