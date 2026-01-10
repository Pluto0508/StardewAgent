import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import heapq
import random

class PPOTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 env: object,
                 buffer: object,  # 缓冲区作为参数输入
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 ppo_epochs: int = 4,
                 batch_size: int = 64,
                 gae_lambda: float = 0.95):
        """
        PPO训练器
        Args:
            model: 策略-价值网络
            env: 强化学习环境
            buffer: 经验回放缓冲区
            learning_rate: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO裁剪参数
            ppo_epochs: PPO更新轮次
            batch_size: 批次大小
            gae_lambda: GAE优势估计参数
        """
        self.model = model
        self.env = env
        self.buffer = buffer
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # 超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda

    def collect_trajectories(self, num_steps: int):
        """收集轨迹数据"""
        state = self.env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_dist, value = self.model(state_tensor)
            
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            next_state, reward, done, _ = self.env.step(action.item())
            
            # 将转移数据存入缓冲区
            self.buffer.add(state, action.item(), log_prob.item(), reward, value.item(), done)
            
            state = next_state if not done else self.env.reset()

    def compute_advantages(self, rewards, values, dones):
        """计算GAE优势函数"""
        advantages = []
        returns = []
        last_advantage = 0
        
        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return torch.tensor(advantages), torch.tensor(returns)

    def update(self):
        """执行PPO更新"""
        # 从缓冲区采样
        data = self.buffer.sample(self.batch_size)
        if data is None:
            return
        
        indices, weights, states, actions, old_log_probs, rewards, values, dones = data
        
        # 计算优势函数
        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        
        # 计算新策略的概率
        action_dist, new_values = self.model(states)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # 计算策略比率
        ratio = (new_log_probs - old_log_probs).exp()
        
        # 计算策略损失 (带裁剪)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失 (使用重要性采样权重)
        value_loss = 0.5 * (weights * (new_values.squeeze() - returns).pow(2)).mean()
        
        # 总损失
        loss = policy_loss + value_loss - 0.01 * entropy
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # 计算TD误差并更新优先级
        with torch.no_grad():
            td_errors = (returns - new_values.squeeze()).abs().numpy()
            self.buffer.update_priorities(indices, td_errors)

    def train(self, total_timesteps: int):
        """主训练循环"""
        num_steps_per_update = 2048  # 每次更新收集的步数
        
        for update in range(total_timesteps // num_steps_per_update):
            self.collect_trajectories(num_steps_per_update)
            for _ in range(self.ppo_epochs):
                self.update()
            print(f"Update {update+1}: Policy updated with PER")