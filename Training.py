import gymnasium as gym
import minigrid
import babyai
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from Reward_Algorithm import *
from Grid_Maze import *

class ImprovedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape  
        in_channels = obs_shape[0]           
        H, W = obs_shape[1], obs_shape[2]   

        self.obj_stream = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        self.coord_stream = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        cx, cy = W // 2, H // 2
        x_coords = (torch.arange(W).float() - cx) / max(cx, 1)
        y_coords = (torch.arange(H).float() - cy) / max(cy, 1)
        x_grid = x_coords.unsqueeze(0).expand(H, -1)
        y_grid = y_coords.unsqueeze(1).expand(-1, W)
        coord_grid = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
        self.register_buffer('coord_grid', coord_grid)

        with torch.no_grad():
            dummy_obj   = torch.zeros(1, in_channels, H, W)
            dummy_coord = torch.zeros(1, 2, H, W)
            n_obj   = self.obj_stream(dummy_obj).shape[1]
            n_coord = self.coord_stream(dummy_coord).shape[1]

        self.fusion = nn.Sequential(
            nn.Linear(n_obj + n_coord, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()
        )

        self._apply_init()

    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        obs = observations.float() / 10.0                       
        B = obs.shape[0]
        coords = self.coord_grid.expand(B, -1, -1, -1)            
        obj_feat   = self.obj_stream(obs)                         
        coord_feat = self.coord_stream(coords)                    

        combined = torch.cat([obj_feat, coord_feat], dim=1)       
        return self.fusion(combined)

if "BabyAI-CleanTask-v0" not in gym.envs.registration.registry:
    gym.register(
        id='BabyAI-CleanTask-v0',
        entry_point=CleanTaskEnv,
    )

class CoveringMultiMapWrapper(gym.Wrapper):
    """
    无限新地图训练：
    - 每个Set = 固定地图 + 固定spawn点 + N步
    - 当前地图所有spawn点用完后，切换到全新地图（新seed）
    """
    def __init__(self, env, actions_per_set=30, base_seed=1000):
        super().__init__(env)
        self.actions_per_set = actions_per_set
        self.base_seed = base_seed

        self.steps_in_current_set = 0
        self.current_spawn_idx = 0
        self.global_map_counter = 0          

        self.current_map_seed = None
        self.current_spawn_sequence = []

        self.set_counter = 0

        self._load_new_map()

    def _load_new_map(self):
        """加载一张全新的地图并生成spawn序列"""
        self.current_map_seed = self.base_seed + self.global_map_counter

        self.env.reset(seed=self.current_map_seed)

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        spawn_points = []
        for x in range(1, base_env.width - 1):
            for y in range(1, base_env.height - 1):
                cell = base_env.grid.get(x, y)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    for direction in range(4):
                        spawn_points.append((x, y, direction))

        rng = np.random.RandomState(self.current_map_seed)
        rng.shuffle(spawn_points)

        self.current_spawn_sequence = spawn_points
        self.current_spawn_idx = 0

        print(f"\n📍 加载新地图 #{self.global_map_counter} "
              f"(seed={self.current_map_seed}): {len(spawn_points)} spawn点")

        self.global_map_counter += 1

    def reset(self, **kwargs):
        """每次reset = 开始新的Set"""

        if self.current_spawn_idx >= len(self.current_spawn_sequence):
            self._load_new_map()

        pos_x, pos_y, direction = self.current_spawn_sequence[self.current_spawn_idx]

        obs, info = self.env.reset(seed=self.current_map_seed)

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        base_env.agent_pos = np.array([pos_x, pos_y])
        base_env.agent_dir = direction

        raw_obs = base_env.gen_obs()
        obs = self.env.observation(raw_obs)

        self.steps_in_current_set = 0
        self.set_counter += 1

        info['map_seed'] = self.current_map_seed
        info['map_number'] = self.global_map_counter - 1
        info['spawn_idx'] = self.current_spawn_idx
        info['set_number'] = self.set_counter

        if self.set_counter % 100 == 0:
            total_spawns = len(self.current_spawn_sequence)
            print(f"  完成 {self.set_counter} Sets | "
                f"地图 #{self.global_map_counter - 1} | "
                f"Spawn {self.current_spawn_idx}/{total_spawns}")

        return obs, info

    def step(self, action):
        """执行一步动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps_in_current_set += 1

        if self.steps_in_current_set >= self.actions_per_set:
            self.current_spawn_idx += 1
            truncated = True
            info['set_complete'] = True

        return obs, reward, terminated, truncated, info

class EmpiricalIDCallback(BaseCallback):
    """
    回调函数：定期打印Empirical ID的统计信息
    """
    def __init__(self, verbose=0, print_freq=5000):
        super().__init__(verbose)
        self.print_freq = print_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                ep_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                
                print(f"\n{'='*60}")
                print(f"Step {self.n_calls}")
                print(f"  Episode reward: {ep_rew:.2f}")
                print(f"  Episode length: {ep_len:.1f}")
                
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    if hasattr(env, 'get_stats'):
                        stats = env.get_stats()
                        print(f"  ID stats:")
                        print(f"    Unique states: {stats.get('num_states', 0)}")
                        print(f"    Avg ID reward: {stats.get('avg_intrinsic_reward', 0):.3f}")
                        
                        for key in ['avg_freedom', 'avg_movement', 'avg_position_novelty', 'avg_global_novelty']:
                            if key in stats:
                                print(f"    {key}: {stats[key]:.3f}")
                        break
                    env = env.env
                
                print(f"{'='*60}")
        
        return True

def make_env(room_size, num_walls, num_pockets, use_control_reward=False,
             control_scale=0.1, actions_per_set=30, base_seed=1000):

    def _init():
        env = gym.make(
            "BabyAI-CleanTask-v0",
            room_size=room_size,
            num_walls=num_walls,
            num_pockets=num_pockets,
            use_distance_reward=False,
            has_goal=False,
            render_mode="rgb_array"
        )

        env = ImgObsWrapper(env)

        env = CoveringMultiMapWrapper(
            env,
            actions_per_set=actions_per_set,
            base_seed=base_seed
        )

        if use_control_reward:
            env = EmpiricalInstrumentalDivergenceWrapper(
                env,
                scale=control_scale,
            )

        env = Monitor(env)

        return env

    return _init

def train_with_curriculum(mode, control_scale=2.0):

    print(f"正在启动 {mode.upper()} 模式...")

    curriculum = [{
        "room_size": 11,
        "num_walls": 22,
        "num_pockets": 10,
        "steps": 600000,
        "actions_per_set": 20,
        "base_seed": 1000
    }]

    model = None

    for i, stage in enumerate(curriculum):
        use_control = (mode == "control")
        env = DummyVecEnv([
            make_env(
                stage['room_size'],
                stage['num_walls'],
                stage['num_pockets'],
                use_control_reward=use_control,
                control_scale=control_scale,
                actions_per_set=stage['actions_per_set'],
                base_seed=stage['base_seed']
            )
        ])

        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=False,
            gamma=0.99,
        )

        if model is None:
            policy_kwargs = dict(
                features_extractor_class=ImprovedCNN,
                features_extractor_kwargs=dict(features_dim=256),
                share_features_extractor=False
            )

            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=2e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.15,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=None,
                verbose=1,
                tensorboard_log=f"./logs_multimap_{mode}/"
            )

        callbacks = [
            CheckpointCallback(
                save_freq=50000,
                save_path=f'./checkpoints_{mode}_multimap/',
                name_prefix='model'
            )
        ]

        if mode == "control":
            callbacks.append(EmpiricalIDCallback(print_freq=10000))

        print(f"🎓 开始训练...\n")
        model.learn(
            total_timesteps=stage['steps'], 
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False
        )

    model.save(f"model_{mode}_multimap_final")
    print(f"\n训练完成！")
    print(f"最终模型已保存: model_{mode}_multimap_final.zip\n")

    return model

if __name__ == "__main__":
    train_with_curriculum("control")
    train_with_curriculum("baseline")