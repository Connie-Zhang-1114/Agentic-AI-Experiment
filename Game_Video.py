import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from minigrid.wrappers import FlatObsWrapper
from Training import *

def record_video(model_path, output_path="eval_video.mp4", fps=20):
    print(f"🎬 正在手动录制模型: {model_path}")
    
    # 1. 创建环境 (确保开启 rgb_array)
    env = CleanTaskEnv(room_size=11, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    
    # 2. 加载模型
    model = PPO.load(model_path)
    
    frames = []
    obs, _ = env.reset()
    
    # 3. 运行并捕获每一帧
    terminated = False
    truncated = False
    step_count = 0
    max_steps = 300 # 录制时长
    
    print("🎥 正在渲染帧...")
    while not (terminated or truncated) and step_count < max_steps:
        # 获取当前画面的像素阵列
        frame = env.render()
        frames.append(frame)
        
        # 预测动作 (注意：手动增加 Batch 维度)
        action, _ = model.predict(obs[np.newaxis, ...], deterministic=True)
        
        # 执行
        obs, reward, terminated, truncated, info = env.step(action[0])
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"已录制 {step_count} 帧...")

    # 4. 使用 imageio 保存视频
    print(f"💾 正在合成视频，共 {len(frames)} 帧...")
    imageio.mimsave(output_path, [np.array(f) for f in frames], fps=fps)
    
    env.close()
    print(f"✅ 视频已成功保存至: {output_path}")

# 执行录制
# 确保你的路径正确
if __name__ == "__main__":
    record_video("model_control_multimap_final_fixed.zip")