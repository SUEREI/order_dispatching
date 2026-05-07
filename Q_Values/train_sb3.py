#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Stable Baselines3 训练订单分派模型

支持算法:
- PPO (Proximal Policy Optimization): 适合连续和离散动作空间
- A2C (Advantage Actor-Critic): PPO的简化版本，训练更快但稳定性稍差
- DQN (Deep Q-Network): 适合离散动作空间
- SAC (Soft Actor-Critic): 适合连续动作空间
- TD3 (Twin Delayed DDPG): 适合连续动作空间

推荐使用 PPO，因为它对超参数不敏感，训练稳定。
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

from order_dispatch_env import OrderDispatchEnv

logger = logging.getLogger(__name__)


def train_with_ppo(
    env: OrderDispatchEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    tensorboard_log: Optional[str] = None,
    save_path: str = "models/ppo_order_dispatch",
    verbose: int = 1,
) -> Any:
    """
    使用 PPO 算法训练
    
    Args:
        env: 订单分派环境
        total_timesteps: 总训练步数
        learning_rate: 学习率
        n_steps: 每次更新收集的步数
        batch_size: 批大小
        n_epochs: 每次更新的epoch数
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        clip_range: PPO裁剪范围
        ent_coef: 熵系数
        vf_coef: 值函数系数
        max_grad_norm: 梯度裁剪
        tensorboard_log: TensorBoard日志目录
        save_path: 模型保存路径
        verbose: 日志详细程度
    
    Returns:
        训练好的模型
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    
    logger.info("使用 PPO 算法训练...")
    
    # 创建模型
    model = PPO(
        "MultiInputPolicy",  # 支持字典观测空间
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    
    # 设置回调
    callbacks = []
    
    # 检查点回调
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(total_timesteps // 10, 1000),
            save_path=os.path.dirname(save_path),
            name_prefix="ppo_checkpoint",
        )
        callbacks.append(checkpoint_callback)
    
    # 训练
    logger.info(f"开始训练，总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=True,
    )
    
    # 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}")
    
    return model


def train_with_a2c(
    env: OrderDispatchEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 7e-4,
    n_steps: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 1.0,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    tensorboard_log: Optional[str] = None,
    save_path: str = "models/a2c_order_dispatch",
    verbose: int = 1,
) -> Any:
    """
    使用 A2C 算法训练
    
    A2C 是 PPO 的简化版本，训练速度更快但稳定性稍差。
    """
    from stable_baselines3 import A2C
    
    logger.info("使用 A2C 算法训练...")
    
    model = A2C(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    
    logger.info(f"开始训练，总步数: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}")
    
    return model


def train_with_dqn(
    env: OrderDispatchEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    tensorboard_log: Optional[str] = None,
    save_path: str = "models/dqn_order_dispatch",
    verbose: int = 1,
) -> Any:
    """
    使用 DQN 算法训练
    
    DQN 适合离散动作空间，使用经验回放和目标网络。
    """
    from stable_baselines3 import DQN
    
    logger.info("使用 DQN 算法训练...")
    
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    
    logger.info(f"开始训练，总步数: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"模型已保存至: {save_path}")
    
    return model


def evaluate_model(
    model: Any,
    env: OrderDispatchEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        env: 环境
        n_episodes: 评估的episode数
    
    Returns:
        评估统计信息
    """
    from stable_baselines3.common.evaluation import evaluate_policy
    
    logger.info(f"评估模型，共 {n_episodes} 个episode...")
    
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=True
    )
    
    logger.info(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }


def run_trained_model(
    model: Any,
    env: OrderDispatchEnv,
    n_episodes: int = 1,
    render: bool = True,
) -> None:
    """
    运行训练好的模型
    
    Args:
        model: 训练好的模型
        env: 环境
        n_episodes: 运行的episode数
        render: 是否渲染
    """
    logger.info(f"运行模型，共 {n_episodes} 个episode...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            if render and step_count % 100 == 0:
                print(f"Episode {ep+1}, Step {step_count}, Reward: {total_reward:.2f}")
        
        print(f"\nEpisode {ep+1} 完成:")
        print(f"  总步数: {step_count}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  完成订单: {info.get('completed_orders', 0)}")
        print(f"  准时订单: {info.get('on_time_orders', 0)}")
        print(f"  超时订单: {info.get('overdue_orders', 0)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 Stable Baselines3 训练订单分派模型")
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn"],
        help="强化学习算法 (default: ppo)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="总训练步数 (default: 100000)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="Q_Values/Data/Data_Processed/cleaned_waybill_info.csv",
        help="订单数据路径"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="Q_Values/simulation/models",
        help="模型保存目录"
    )
    parser.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="TensorBoard日志目录"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="训练后评估模型"
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="评估的episode数"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="评估时渲染环境"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    # 创建环境
    logger.info("创建环境...")
    env = OrderDispatchEnv(
        order_data_path=args.data,
        state_dim=27,
        action_dim=5,
        max_couriers=50,
        max_orders=20,
        render_mode="console" if args.render else None,
    )
    
    # 训练
    save_path = os.path.join(args.save_dir, f"{args.algo}_order_dispatch")
    
    if args.algo == "ppo":
        model = train_with_ppo(
            env,
            total_timesteps=args.timesteps,
            tensorboard_log=args.tensorboard,
            save_path=save_path,
        )
    elif args.algo == "a2c":
        model = train_with_a2c(
            env,
            total_timesteps=args.timesteps,
            tensorboard_log=args.tensorboard,
            save_path=save_path,
        )
    elif args.algo == "dqn":
        model = train_with_dqn(
            env,
            total_timesteps=args.timesteps,
            tensorboard_log=args.tensorboard,
            save_path=save_path,
        )
    
    # 评估
    if args.eval:
        eval_results = evaluate_model(model, env, n_episodes=args.n_eval_episodes)
        logger.info(f"评估结果: {eval_results}")
        
        # 运行演示
        run_trained_model(model, env, n_episodes=1, render=args.render)
    
    logger.info("完成！")


if __name__ == "__main__":
    main()
