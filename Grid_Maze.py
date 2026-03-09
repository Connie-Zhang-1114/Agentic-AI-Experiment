import gymnasium as gym
import minigrid
import babyai
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from Grid_Maze import *
from gymnasium import spaces
from minigrid.wrappers import FlatObsWrapper
from minigrid.core.world_object import Wall, Ball, WorldObj
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

class ImgObsWrapper(gym.ObservationWrapper):
    """
    自制 ImgObsWrapper:
    将字典形式的 obs {"image": [...], ...} 转换为纯数组
    兼容自定义的360度观测（形状由环境决定）
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.spaces["image"].shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs["image"], (2, 0, 1))


class MultiWallCell(WorldObj):
    """
    支持多个方向墙的Cell
    可以在同一个格子的不同边缘放置墙，形成"口袋"结构
    """
    def __init__(self, color='grey'):
        super().__init__('wall', color)
        self.walls = {
            'top': False,
            'bottom': False,
            'left': False,
            'right': False
        }

    def add_wall(self, edge):
        if edge in self.walls:
            self.walls[edge] = True
        return self

    def has_wall(self, edge):
        return self.walls.get(edge, False)

    def can_overlap(self):
        return True

    def see_behind(self):
        return False

    def encode(self):
        state = (
            (1 if self.walls['top'] else 0) |
            (2 if self.walls['bottom'] else 0) |
            (4 if self.walls['left'] else 0) |
            (8 if self.walls['right'] else 0)
        )
        return (
            OBJECT_TO_IDX['wall'],
            COLOR_TO_IDX.get(self.color, 0),
            state
        )

    def render(self, img):
        tile_size = img.shape[0]
        wall_thickness = 8
        wall_color = np.array(COLORS[self.color])
        if self.walls['top']:
            img[0:wall_thickness, :, :] = wall_color
        if self.walls['bottom']:
            img[tile_size - wall_thickness:tile_size, :, :] = wall_color
        if self.walls['left']:
            img[:, 0:wall_thickness, :] = wall_color
        if self.walls['right']:
            img[:, tile_size - wall_thickness:tile_size, :] = wall_color

    def __repr__(self):
        walls = ''.join([
            'T' if self.walls['top'] else '-',
            'B' if self.walls['bottom'] else '-',
            'L' if self.walls['left'] else '-',
            'R' if self.walls['right'] else '-'
        ])
        return f"MultiWall({self.color}, [{walls}])"

class CleanTaskEnv(MiniGridEnv):
    def __init__(self, room_size=15, num_walls=20, num_pockets=4,
                 use_distance_reward=False, has_goal=False,
                 view_radius=3, **kwargs):
        """
        view_radius: agent能看到的半径，默认None=覆盖整张地图
                     实际视野 = (2*view_radius+1) x (2*view_radius+1) 的方形区域
                     无墙方向一望到底，有墙方向视觉遮挡
        """
        self.num_walls = num_walls
        self.num_pockets = num_pockets
        self.use_distance_reward = use_distance_reward
        self.has_goal = has_goal
        self.view_radius = view_radius if view_radius is not None else room_size
        self.visit_counts = defaultdict(int)
        self.last_distance = None
        mission_space = MissionSpace(mission_func=lambda: "go to the red ball")
        self.min_distance_ever = None
        self.position_history = []
        self.recent_distances = []

        super().__init__(
            mission_space=mission_space,
            grid_size=room_size,
            max_steps=1000,
            see_through_walls=False,  
            **kwargs
        )

        # 动作空间：左转/右转/前进/后退
        self.action_space = spaces.Discrete(4)
        self.actions = type('obj', (object,), {
            'left': 0,
            'right': 1,
            'forward': 2,
            'backward': 3,
        })()

        # 重新定义observation_space为360度方形局部视野
        view_size = 2 * self.view_radius + 1
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "image": spaces.Box(
                low=0, high=255,
                shape=(view_size, view_size, 3),
                dtype=np.uint8
            )
        })

    # ===========================
    # 360度局部观测（核心改动）
    # ===========================

    def _ray_visible(self, x0, y0, x1, y1):
        """
        Bresenham光线投射：从(x0,y0)射向(x1,y1)
        规则：遇到不透明格子，该格子本身可见，其后面的格子不可见
        """
        cells = self._bresenham_line(x0, y0, x1, y1)
        for idx, (cx, cy) in enumerate(cells[1:], start=1):
            is_target = (idx == len(cells) - 1)

            # 超出地图范围：不可见
            if not (0 <= cx < self.width and 0 <= cy < self.height):
                return False

            cell = self.grid.get(cx, cy)
            is_opaque = (cell is not None and not self._cell_transparent(cell))

            if is_opaque:
                # 不透明格子本身可见（能看到墙），但墙后面不可见
                return is_target

        return True

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham直线算法，返回路径上所有格子"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def _cell_transparent(self, cell):
        """判断格子是否允许光线穿过"""
        if isinstance(cell, MultiWallCell):
            # MultiWallCell：有任何墙面就不透明（光线不能穿过有墙的格子）
            return not any(cell.walls.values())
        if hasattr(cell, 'see_behind'):
            return cell.see_behind()
        return True

    def _compute_visible_cells(self):
        """
        以agent为中心，对视野内每个格子做光线投射
        返回可见格子集合 {(x, y)}
        """
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        r = self.view_radius
        visible = set()
        visible.add((ax, ay))

        for tx in range(ax - r, ax + r + 1):
            for ty in range(ay - r, ay + r + 1):
                if tx == ax and ty == ay:
                    continue
                if self._ray_visible(ax, ay, tx, ty):
                    visible.add((tx, ty))

        return visible

    def gen_obs(self):
        """
        覆盖默认gen_obs：生成以agent为中心的360度局部观测
        视野 = (2*view_radius+1) x (2*view_radius+1)
        不可见格子编码为 (0, 0, 0)
        """
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        r = self.view_radius
        view_size = 2 * r + 1

        visible_cells = self._compute_visible_cells()

        obs_image = np.zeros((view_size, view_size, 3), dtype=np.uint8)

        for i, gx in enumerate(range(ax - r, ax + r + 1)):
            for j, gy in enumerate(range(ay - r, ay + r + 1)):

                # Agent自身位置
                if gx == ax and gy == ay:
                    obs_image[j, i] = np.array([
                        OBJECT_TO_IDX['agent'],
                        COLOR_TO_IDX.get('red', 0),
                        self.agent_dir   # 朝向编码在第3通道
                    ], dtype=np.uint8)
                    continue

                # 不可见格子：保持 (0,0,0)
                if (gx, gy) not in visible_cells:
                    continue

                # 地图边界外：编码为实心墙
                if not (0 <= gx < self.width and 0 <= gy < self.height):
                    obs_image[j, i] = np.array([
                        OBJECT_TO_IDX['wall'],
                        COLOR_TO_IDX.get('grey', 0),
                        0
                    ], dtype=np.uint8)
                    continue

                cell = self.grid.get(gx, gy)
                if cell is None:
                    obs_image[j, i] = np.array([
                        OBJECT_TO_IDX['empty'],
                        0, 0
                    ], dtype=np.uint8)
                else:
                    encoded = cell.encode()
                    obs_image[j, i] = np.array(encoded, dtype=np.uint8)

        return {
            "image": obs_image,
            "direction": self.agent_dir,
            "mission": self.mission,
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.episode_visit_counts = defaultdict(int)
        self.position_history = [tuple(self.agent_pos)]

        current_pos = tuple(self.agent_pos)
        self.episode_visit_counts[current_pos] = 1

        total_cells = (self.grid.width - 2) * (self.grid.height - 2)
        info["coverage"] = (1 / total_cells) * 100
        info["unique_visited"] = 1
        info["agent_pos"] = tuple(self.agent_pos)

        if self.has_goal:
            self.last_distance = self._manhattan_distance(
                self.agent_pos,
                self.red_ball.cur_pos
            )
            self.min_distance_ever = self.last_distance
            self.recent_distances = [self.last_distance]
        else:
            self.last_distance = None
            self.min_distance_ever = None
            self.recent_distances = []

        return obs, info

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_open_cell(self, x, y):
        if x <= 0 or x >= self.width - 1 or y <= 0 or y >= self.height - 1:
            return False
        cell = self.grid.get(x, y)
        if isinstance(cell, MultiWallCell):
            if any(cell.walls.values()):
                return False
        if cell is not None and not isinstance(cell, MultiWallCell):
            if cell.type == 'wall':
                return False
        directions = [
            ('top', 0, -1), ('bottom', 0, 1),
            ('left', -1, 0), ('right', 1, 0)
        ]
        for direction, dx, dy in directions:
            neighbor_cell = self.grid.get(x + dx, y + dy)
            if isinstance(neighbor_cell, MultiWallCell):
                opposite = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
                if neighbor_cell.has_wall(opposite[direction]):
                    return False
        return True

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        edges = ['top', 'bottom', 'left', 'right']
        walls_placed = 0
        while walls_placed < self.num_walls:
            pos = (self._rand_int(1, width - 1), self._rand_int(1, height - 1))
            cell = self.grid.get(*pos)
            if cell is None:
                cell = MultiWallCell()
                self.grid.set(pos[0], pos[1], cell)
            if isinstance(cell, MultiWallCell):
                edge_type = self._rand_elem(edges)
                if not cell.has_wall(edge_type):
                    cell.add_wall(edge_type)
                    walls_placed += 1

        pockets_created = 0
        while pockets_created < self.num_pockets:
            pos = (self._rand_int(1, width - 1), self._rand_int(1, height - 1))
            cell = self.grid.get(*pos)
            if cell is None:
                pocket = MultiWallCell()
                num_walls_in_pocket = self._rand_int(2, 4)
                selected_edges = self._rand_subset(edges, num_walls_in_pocket)
                for edge in selected_edges:
                    pocket.add_wall(edge)
                self.grid.set(pos[0], pos[1], pocket)
                pockets_created += 1

        if self.has_goal:
            self.red_ball = Ball('red')
            self.red_ball.can_overlap = lambda: True
            self.place_obj(self.red_ball)
        else:
            self.red_ball = None

        self.place_agent()

    def step(self, action):
        old_distance = None
        if self.use_distance_reward and self.has_goal:
            old_distance = self._manhattan_distance(
                self.agent_pos,
                self.red_ball.cur_pos
            )

        # 后退动作单独处理
        if action == self.actions.backward:
            obs, reward, terminated, truncated, info = self._step_backward()

        # 前进被墙挡：原有逻辑
        elif action == self.actions.forward and not self._can_move_forward():
            self.step_count += 1
            obs = self.gen_obs()
            current_pos = tuple(self.agent_pos)
            if current_pos not in self.episode_visit_counts:
                self.episode_visit_counts[current_pos] = 0
            unique_visited = len(self.episode_visit_counts)
            total_cells = (self.grid.width - 2) * (self.grid.height - 2)
            info = {
                "coverage": (unique_visited / total_cells) * 100,
                "unique_visited": unique_visited,
                "agent_pos": current_pos
            }
            return obs, 0.0, False, self.step_count >= self.max_steps, info

        else:
            # 左转/右转/前进走super().step()，然后用自定义obs覆盖
            _, _, terminated, truncated, info = super().step(action)
            obs = self.gen_obs()

        current_pos = tuple(self.agent_pos)
        self.position_history.append(current_pos)
        self.episode_visit_counts[current_pos] += 1

        unique_visited = len(self.episode_visit_counts)
        total_cells = (self.grid.width - 2) * (self.grid.height - 2)
        info["coverage"] = (unique_visited / total_cells) * 100
        info["unique_visited"] = unique_visited
        info["agent_pos"] = current_pos

        reward = 0.0

        if self.has_goal:
            new_distance = self._manhattan_distance(
                self.agent_pos,
                self.red_ball.cur_pos
            )
            if np.array_equal(self.agent_pos, self.red_ball.cur_pos):
                reward = 20.0
                terminated = True
            elif self.use_distance_reward and old_distance is not None:
                distance_change = old_distance - new_distance
                reward = distance_change * 2

            info["distance"] = new_distance
            info["distance_reward"] = reward

        return obs, reward, terminated, truncated, info

    def _step_backward(self):
        """后退：朝当前朝向的反方向移动一格，朝向不变"""
        # agent_dir: 0=右, 1=下, 2=左, 3=上
        reverse_dir = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
        dx, dy = reverse_dir[self.agent_dir]
        back_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        self.step_count += 1

        if self._can_move_to(back_pos, moving_backward=True):
            self.agent_pos = np.array(back_pos)

        obs = self.gen_obs()
        truncated = self.step_count >= self.max_steps
        info = {"agent_pos": tuple(self.agent_pos)}

        return obs, 0.0, False, truncated, info

    def _can_move_to(self, target_pos, moving_backward=False):
        """检查能否移动到目标位置（支持MultiWallCell方向检查）"""
        tx, ty = int(target_pos[0]), int(target_pos[1])
        if not (0 < tx < self.width - 1 and 0 < ty < self.height - 1):
            return False

        current_cell = self.grid.get(*self.agent_pos)
        target_cell = self.grid.get(tx, ty)

        direction_map = {0: 'right', 1: 'bottom', 2: 'left', 3: 'top'}
        opposite_map = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}

        if moving_backward:
            moving_direction = opposite_map[direction_map[self.agent_dir]]
        else:
            moving_direction = direction_map[self.agent_dir]

        exit_direction = moving_direction
        entry_direction = opposite_map[moving_direction]

        if isinstance(current_cell, MultiWallCell):
            if current_cell.has_wall(exit_direction):
                return False

        if isinstance(target_cell, MultiWallCell):
            if target_cell.has_wall(entry_direction):
                return False

        if target_cell is not None and not isinstance(target_cell, MultiWallCell):
            if not target_cell.can_overlap():
                return False

        return True

    def _can_move_forward(self):
        return self._can_move_to(tuple(self.front_pos), moving_backward=False)
