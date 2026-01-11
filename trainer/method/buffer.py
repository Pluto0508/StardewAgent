import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import heapq
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, log_prob, reward, value, done):
        """添加经验样本"""
        # 新样本的初始优先级设为当前最大优先级
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, log_prob, reward, value, done))
        else:
            self.buffer[self.position] = (state, action, log_prob, reward, value, done)
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """采样一个batch的经验"""
        if self.size == 0:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 根据优先级采样
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 解包样本
        states, actions, log_probs, rewards, values, dones = zip(*samples)
        
        return (indices, 
                weights,
                np.array(states),
                np.array(actions),
                np.array(log_probs),
                np.array(rewards),
                np.array(values),
                np.array(dones))

    def update_priorities(self, indices, priorities):
        """更新样本优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0