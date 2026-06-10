import gymnasium as gym
import minigrid
import babyai
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from minigrid.core.world_object import Ball, WorldObj
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

class ImgObsWrapper(gym.ObservationWrapper):
    """
    Converts the default dict observation {"image": [...], ...} into a
    plain numpy array with shape (C, H, W), compatible with CNN-based policies.
    Supports the customized 4-channel 360-degree observation format.
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
    A grid cell supporting directional walls on any of its four edges.
    Multiple walls on the same cell create pocket structures that restrict
    agent movement and reduce directional freedom.
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
        """Activate a wall on the specified side (top/bottom/left/right)."""
        if edge in self.walls:
            self.walls[edge] = True
        return self

    def has_wall(self, edge):
        """Return True if this cell has a wall on the given side."""
        return self.walls.get(edge, False)

    def can_overlap(self):
        """Allow the agent to occupy this cell despite wall is present."""
        return True

    def see_behind(self):
        """Block visibility through this cell."""
        return False

    def encode(self):
        """
        Encode wall configuration as a bitmask in the third channel.
        Bits: top=1, bottom=2, left=4, right=8.
        """
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
        """Present walls as colored strips on the corresponding sides."""
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
    """
    Custom MiniGrid environment with:
    - 4-action space: turn left, turn right, move forward, move backward
    - 4-channel 360-degree local observation (7x7 square vision)
    - MultiWallCell obstacles that create pockets with low directional freedom
    - Optional red ball goal with sparse reward (+255 on reach)
    """
    def __init__(self, room_size=15, num_walls=20, num_pockets=4,
                 use_distance_reward=False, has_goal=False,**kwargs):
        self.num_walls = num_walls
        self.num_pockets = num_pockets
        self.use_distance_reward = use_distance_reward
        self.has_goal = has_goal
        mission_space = MissionSpace(mission_func=lambda: "go to the red ball")
        self.position_history = []

        super().__init__(
            mission_space=mission_space,
            grid_size=room_size,
            max_steps=2000,
            see_through_walls=False,    
            **kwargs
        )

        # 4-action space: turn left, turn right, forward, backward
        self.action_space = spaces.Discrete(4)
        self.actions = type('obj', (object,), {
            'left': 0,
            'right': 1,
            'forward': 2,
            'backward': 3,
        })()

        # Redefine observation space as 4-channel 360-degree view
        vs = self.agent_view_size
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "image": spaces.Box(
                low=0, high=255,
                shape=(vs, vs, 4),
                dtype=np.uint8
            )
        })

    def _compute_directional_freedom(self, position):
        """
        Compute the directional freedom of a cell as a value in [0, 1].
        Counts how many directions are movable (considering both exit walls on the current cell and entry walls on
        neighboring cells). Returns accessible_count / 4.0.
        """
        direction_map = {
        (1,  0): ('right', 'left'),
        (-1, 0): ('left',  'right'),
        (0,  1): ('bottom', 'top'),
        (0, -1): ('top',   'bottom'),
        }

        accessible = 0
        for (dx, dy), (exit_dir, entry_dir) in direction_map.items():
            target = (position[0] + dx, position[1] + dy)

            if not (0 < target[0] < self.width - 1 and
                0 < target[1] < self.height - 1):
                continue

            current_cell = self.grid.get(*position)
            if isinstance(current_cell, MultiWallCell):
                if current_cell.has_wall(exit_dir):
                    continue

            target_cell = self.grid.get(*target)
            if target_cell is not None:
                if target_cell.type == 'wall' and not isinstance(target_cell, MultiWallCell):
                    continue
                if isinstance(target_cell, MultiWallCell):
                    if target_cell.has_wall(entry_dir):
                        continue

            accessible += 1

        return accessible / 4.0

    def gen_obs(self):
        """
        Generate a 4-channel 360-degree observation centered on the agent.
        Channels: [object_type, color, wall_state, freedom_value].
        The red ball cell is encoded with freedom=255 as the biggest signal.
        """
        vs = self.agent_view_size  
        r = vs // 2                
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])

        obs_image = np.zeros((vs, vs, 4), dtype=np.uint8) 
        original_dir = self.agent_dir

        dirs_to_sample = [original_dir, (original_dir + 2) % 4]

        # Sample from current direction and opposite direction for 360 coverage
        for sample_dir in dirs_to_sample:
            self.agent_dir = sample_dir
            grid, vis_mask = self.gen_obs_grid()
            self.agent_dir = original_dir

            for gx in range(ax - r, ax + r + 1):
                for gy in range(ay - r, ay + r + 1):
                    if gx == ax and gy == ay:
                        continue

                    self.agent_dir = sample_dir
                    vx, vy = self.get_view_coords(gx, gy)
                    self.agent_dir = original_dir

                    if vx < 0 or vy < 0 or vx >= vs or vy >= vs:
                        continue
                    if not vis_mask[vx, vy]:
                        continue

                    self.agent_dir = original_dir
                    ox, oy = self.get_view_coords(gx, gy)
                    self.agent_dir = original_dir

                    oy_out = oy - r
                    ox_out = ox

                    if oy_out < 0:
                        oy_out = vs + oy_out

                    if oy_out < 0 or oy_out >= vs or ox_out < 0 or ox_out >= vs:
                        continue

                    if np.any(obs_image[oy_out, ox_out, :3] != 0):
                        continue

                    cell = grid.get(vx, vy)
                    if cell is None:
                        encoded = np.array([OBJECT_TO_IDX['empty'], 0, 0], dtype=np.uint8)
                    else:
                        encoded = np.array(cell.encode(), dtype=np.uint8)

                    obs_image[oy_out, ox_out, :3] = encoded

                    # # Fourth Channel：red ball=255，other cells=freedom value
                    if 0 < gx < self.width - 1 and 0 < gy < self.height - 1:
                        cell_at_pos = self.grid.get(gx, gy)
                        if cell_at_pos is not None and hasattr(cell_at_pos, 'type') and cell_at_pos.type == 'ball':
                            obs_image[oy_out, ox_out, 3] = 255  
                        else:
                            freedom = self._compute_directional_freedom((gx, gy))
                            obs_image[oy_out, ox_out, 3] = int(freedom)
                
        # Write agent's own cell at center
        obs_image[r, r] = np.array([
        OBJECT_TO_IDX['empty'], 0, 0, 0
        ], dtype=np.uint8)

        return {
        "image": obs_image,
        "direction": original_dir,
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

        return obs, info

    def _gen_grid(self, width, height):
        """
        Generate the grid layout:
        1. Place maze boundary walls.
        2. Randomly create num_walls single-side MultiWallCells.
        3. Randomly create num_pockets multi-sides MultiWallCells.
        4. Place red ball (has_goal=True) and agent at random open positions.
        """
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
        """
        Execute one step. Handles the customized backward action separately.
        Updates visit counts and coverage stats each step.
        If has_goal=True, checks whether agents reach the red ball and assign reward.
        """

        # Handle backward action independently
        if action == self.actions.backward:
            obs, reward, terminated, truncated, info = self._step_backward()

        # Forward blocked by wall: increase step count but stay in place
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
                reward = 255.0
                terminated = True

            info["distance"] = new_distance
            info["distance_reward"] = reward

        return obs, reward, terminated, truncated, info

    def _step_backward(self):
        """
        Move a step in the direction opposite to the agent's facing.
        """
        # agent_dir: 0=right, 1=down, 2=left, 3=up
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
        """
        Check whether the agent can move to target_pos.
        """
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
        """Check whether the agent can move forward."""
        return self._can_move_to(tuple(self.front_pos), moving_backward=False)
