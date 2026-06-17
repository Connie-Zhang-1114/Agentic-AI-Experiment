# Agentic AI Experiment: Instrumental Control as a Reward Surrogate

## Overview

This project investigates whether a reinforcement learning agent that treats
instrumental control (local environmental freedom) as a reward adapts faster 
to reward distribution shifts than an agent trained without this signal. 
The experiment is built on a custom MiniGrid-based environment and 
compares three agents — a **control agent**, a **baseline agent**, and 
a **random agent** — across a two-phase training pipeline.

### Core hypothesis

If an agent is trained to prefer high-controllability regions (cells with
more open directions, fewer walls and pockets), does this exploration bias
help or hurt its ability to adapt when a reward (a red ball) is introduced
in a new location?

## Repository structure

```
Grid_Maze_360_vision.py    Custom MiniGrid environment, observation encoding, grid generation
Reward_Algorithm.py        Freedom-based intrinsic reward wrapper 
Training_360_vision.py     Phase 1 training: exploration-only, no goal
Learning_in_Testing.py     Phase 2 training: goal-directed fine-tuning 
```

## Environment design (`Grid_Maze_360_vision.py`)

### CleanTaskEnv

A custom MiniGrid environment with:
- **4-action space**: turn left, turn right, move forward, move backward
- **4-channel local observation** (7×7 window): object type, color, wall
  state, and a directional freedom value
- **MultiWallCell**: grid cells that support directional walls on any of
  their four edges, used to construct "pockets" — low-freedom regions
  that restrict movement
- **Optional red ball goal**: when `has_goal=True`, a ball is placed on the
  grid; reaching it terminates the episode with reward `+255`

### Directional freedom

The core metric of the experiment. For any grid cell, freedom is defined as:

```
freedom(cell) = accessible_directions / 4.0
```

where a direction is accessible if neither the current cell's exit edge
nor the neighboring cell's entry edge has a wall. This value is encoded
directly into the 4th observation channel, making local controllability
visible to the policy network without requiring it to be inferred from
raw pixel data.

### 360-degree observation

`gen_obs()` samples both the agent's current facing direction and its
opposite, merging the two views into a single 7×7 window. This ensures
full 360-degree awareness within the limited local window.

## Reward design (`Reward_Algorithm.py`)

### EmpiricalInstrumentalDivergenceWrapper

Adds an intrinsic reward to the control agent based on the freedom of the
cell the agent moves into:

```
intrinsic_reward = (freedom ** 2) * scale
total_reward      = env_reward + intrinsic_reward
```

Squaring the freedom value amplifies the preference for highly open cells. 
The wrapper independently recomputes freedom from the grid state 
(rather than reading the observation tensor), so it can be attached or removed 
without changing the environment itself.

## Training pipeline

### Phase 1 — Exploration pretraining (`Training_360_vision.py`)

Both agents are trained from scratch with **no goal present**
(`has_goal=False`) across many generated maps:

- **Baseline agent**: trained with entropy regularization only
  (`ent_coef=0.05`). No intrinsic reward — exploration emerges purely from
  the policy's inherent stochasticity and the entropy bonus that prevents
  early convergence to deterministic behavior.
- **Control agent**: same setup, plus the freedom-based intrinsic reward
  from `EmpiricalInstrumentalDivergenceWrapper`.

`CoveringMultiMapWrapper` ensures the agent trains across a continuous
stream of new maps and spawn points, ensuring it's learning a generalizable exploration policy.

Both agents use the same PPO architecture (`ImprovedCNN` — a dual-stream
CNN combining object features and normalized spatial coordinates) and the
same hyperparameters.

### Phase 2 — Goal-directed fine-tuning (`Learning_in_Testing.py`)

Phase 1 agents freely explores the map without the goal. Phase 2
introduces `has_goal=True` and fine-tunes each agent (loaded from 
Phase 1 checkpoint) on a single fixed map (via `FixedSeedWrapper`) so that
learning curves are comparable across agents.

Three conditions are tested:
- **random**: an initialized PPO model (no Phase 1 pre-training), used as
  a sanity-check
- **baseline**: Phase 1 baseline checkpoint, continued training
- **control**: Phase 1 control checkpoint, continued training 

`SuccessTrackingCallback` records every successful episode (reward ≥ 50)
along with its relative timestep and episode length, enabling step-trend
analysis across training. Results are exported to a formatted Excel
workbook (`export_summary_excel`) summarising success counts and average
steps-to-goal per time window for direct comparison across the three agents.

## Requirements

```
gymnasium
minigrid
stable-baselines3
torch
numpy
openpyxl
```