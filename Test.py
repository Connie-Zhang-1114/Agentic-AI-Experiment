from Training import *
    
def test_model_with_adaptation(model_path, num_trials=500, finetune_steps=1000,
                               test_episodes=1, render=False, mode="baseline",
                               control_scale=1.5):
    """
    Test-time adaptation测试：
    每个trial先在新地图上fine-tune（无goal），再测试（有goal）
    
    Args:
        model_path:      预训练模型路径
        num_trials:      试验次数（每次用不同seed/地图）
        finetune_steps:  每张新地图的fine-tune步数
        test_episodes:   fine-tune后测试的episode数
        render:          是否可视化
        mode:            'baseline' 或 'control'
        control_scale:   control模式的freedom reward scale
    """

    print(f"🧪 Test-time Adaptation 测试 [{mode.upper()}]: {model_path}")
    print(f"{'='*60}")
    print(f"  试验次数:       {num_trials}")
    print(f"  每次fine-tune:  {finetune_steps} steps")
    print(f"  每次测试:       {test_episodes} episodes")
    print(f"{'='*60}\n")

    all_trials_results = []

    for trial_idx in range(num_trials):
        trial_seed = 9000 + trial_idx

        use_control = (mode == "control")

        finetune_env = gym.make(
            "BabyAI-CleanTask-v0",
            room_size=11,
            num_walls=22,
            num_pockets=10,
            use_distance_reward=False,
            has_goal=False,
            render_mode="rgb_array"
        )
        finetune_env = ImgObsWrapper(finetune_env)
        if use_control:
            finetune_env = EmpiricalInstrumentalDivergenceWrapper(
                finetune_env, scale=control_scale
            )
        finetune_env = DummyVecEnv([lambda: finetune_env])
        finetune_env = VecNormalize(finetune_env, norm_obs=False, norm_reward=False)
        finetune_env.seed(trial_seed)

        model = PPO.load(model_path, env=finetune_env)
        model.learn(
            total_timesteps=finetune_steps,
            reset_num_timesteps=True,
            progress_bar=False
        )
        finetune_env.close()

        test_env = gym.make(
            "BabyAI-CleanTask-v0",
            room_size=11,
            num_walls=22,
            num_pockets=10,
            use_distance_reward=False,
            has_goal=True,
            render_mode="human" if render else "rgb_array"
        )
        test_env = ImgObsWrapper(test_env)
        test_env.reset(seed=trial_seed)

        base_env = test_env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        initial_agent_pos = tuple(base_env.agent_pos)
        initial_ball_pos  = tuple(base_env.red_ball.cur_pos) if base_env.red_ball else None

        trial_successes    = 0
        trial_success_steps = []
        trial_rewards      = []
        trial_lengths      = []

        for episode in range(test_episodes):
            obs, _ = test_env.reset(seed=trial_seed)

            episode_reward = 0
            episode_length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=False)
                if isinstance(action, np.ndarray):
                    action = action.item()

                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward
                episode_length += 1

                if render:
                    test_env.render()

            if terminated:
                trial_successes += 1
                trial_success_steps.append(episode_length)

            trial_rewards.append(episode_reward)
            trial_lengths.append(episode_length)

        test_env.close()

        trial_result = {
            'trial_idx':          trial_idx,
            'seed':               trial_seed,
            'success_rate':       trial_successes / test_episodes,
            'successes':          trial_successes,
            'avg_success_steps':  np.mean(trial_success_steps) if trial_success_steps else None,
            'min_success_steps':  min(trial_success_steps) if trial_success_steps else None,
            'max_success_steps':  max(trial_success_steps) if trial_success_steps else None,
            'avg_reward':         np.mean(trial_rewards),
            'avg_steps':          np.mean(trial_lengths),
            'initial_agent_pos':  initial_agent_pos,
            'initial_ball_pos':   initial_ball_pos
        }
        all_trials_results.append(trial_result)

        if (trial_idx + 1) % 50 == 0:
            done = trial_idx + 1
            success_so_far = [r['avg_success_steps'] for r in all_trials_results
                              if r['avg_success_steps'] is not None]
            print(f"  [{mode}] Trial {done}/{num_trials} | "
                  f"Success Rate: {np.mean([r['success_rate'] for r in all_trials_results])*100:.1f}% | "
                  f"Successf Steps mean: {np.mean(success_so_far):.1f}" if success_so_far else
                  f"  [{mode}] Trial {done}/{num_trials} | 暂无成功")

    print(f"\n{'='*60}")
    print(f"总体统计 [{mode.upper()}]")
    print(f"{'='*60}")

    all_success_rates = [r['success_rate'] for r in all_trials_results]
    all_success_steps = [r['avg_success_steps'] for r in all_trials_results
                         if r['avg_success_steps'] is not None]

    print(f"Number of Trials:       {num_trials}")
    print(f"Fine-tune Steps:  {finetune_steps} per trial")
    print(f"\n--- 成功率统计 ---")
    print(f"Success Rate mean:     {np.mean(all_success_rates)*100:.1f}%")
    print(f"Success Rate std:   {np.std(all_success_rates)*100:.1f}%")

    if all_success_steps:
        print(f"\n--- 成功步数统计 ---")
        print(f"Success Steps mean:   {np.mean(all_success_steps):.1f}")
        print(f"Success Steps std: {np.std(all_success_steps):.1f}")
        print(f"Steps Range:       [{np.min(all_success_steps):.1f}, {np.max(all_success_steps):.1f}]")

    print(f"\n--- 每个Trial详情（仅成功） ---")
    print(f"{'Trial':>6} {'Seed':>6} {'Success Rate':>8} {'Success Steps':>10}")
    print(f"{'-'*50}")
    for r in all_trials_results:
        if r['success_rate'] > 0:
            steps_str = f"{r['avg_success_steps']:.1f}" if r['avg_success_steps'] else "N/A"
            print(f"{r['trial_idx']+1:>6} {r['seed']:>6} "
                  f"{r['success_rate']*100:>7.1f}% "
                  f"{steps_str:>10} ")

    print(f"{'='*60}\n")

    return {
        'mode':               mode,
        'num_trials':         num_trials,
        'finetune_steps':     finetune_steps,
        'test_episodes':      test_episodes,
        'trials_results':     all_trials_results,
        'mean_success_rate':  np.mean(all_success_rates),
        'std_success_rate':   np.std(all_success_rates),
        'mean_success_steps': np.mean(all_success_steps) if all_success_steps else None,
        'std_success_steps':  np.std(all_success_steps) if all_success_steps else None,
    }

if __name__ == "__main__":
    test_model_with_adaptation("model_baseline_multimap_final", mode="baseline")
    test_model_with_adaptation("model_control_multimap_final", mode="control")
