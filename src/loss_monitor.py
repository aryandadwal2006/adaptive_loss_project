"""
RL-based Loss Monitor Agent
Learns optimal loss adaptations through reinforcement learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle
import os
from .config import RL_CONFIG, DEVICE, SAFETY_CONFIG

class LossMonitorAgent:
    """
    Reinforcement Learning agent that monitors training dynamics
    and learns optimal loss function adaptations
    """
    
    def __init__(self, 
                 state_dim=None, 
                 action_dim=None, 
                 hidden_dim=None,
                 learning_rate=None,
                 memory_size=None):
        
        # Use config values or defaults
        self.state_dim = state_dim or RL_CONFIG['state_dim']
        self.action_dim = action_dim or RL_CONFIG['action_dim']
        self.hidden_dim = hidden_dim or RL_CONFIG['hidden_dim']
        self.learning_rate = learning_rate or RL_CONFIG['learning_rate']
        self.memory_size = memory_size or RL_CONFIG['memory_size']
        
        # Q-Network for learning action values
        self.q_network = self._build_q_network().to(DEVICE)
        
        # Target network for stable training
        self.target_network = self._build_q_network().to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Exploration parameters
        self.epsilon = RL_CONFIG['epsilon']
        self.epsilon_min = RL_CONFIG['epsilon_min']
        self.epsilon_decay = RL_CONFIG['epsilon_decay']
        
        # Training statistics
        self.total_steps = 0
        self.total_rewards = []
        self.loss_history = []
        
        # Action mapping
        self.action_mapping = self._create_action_mapping()
        
    def _build_q_network(self):
        """Build the Q-network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_dim, self.action_dim)
        )

    
    def _create_action_mapping(self):
        """Create mapping from actions to loss parameters"""
        return {
            0: {  # Conservative adaptation
                'main_weight': 1.0,
                'aux_weight': 0.0,
                'schedule_factor': 1.0,
                'exploration_bonus': 0.0
            },
            1: {  # Moderate adaptation
                'main_weight': 0.8,
                'aux_weight': 0.2,
                'schedule_factor': 1.1,
                'exploration_bonus': 0.05
            },
            2: {  # Aggressive adaptation
                'main_weight': 1.2,
                'aux_weight': 0.1,
                'schedule_factor': 0.9,
                'exploration_bonus': 0.1
            }
        }
    
    def get_state(self, loss_history, gradient_stats, training_info):
        """
        Extract current training state from available information
        
        Args:
            loss_history: List of recent loss values
            gradient_stats: Dictionary with gradient statistics
            training_info: Dictionary with training metadata
        
        Returns:
            torch.Tensor: State representation
        """
        # Current loss
        current_loss = loss_history[-1] if loss_history else 0.0
        
        # Loss trend (average of last 5 values)
        if len(loss_history) >= 5:
            loss_trend = np.mean(loss_history[-5:])
            loss_variance = np.var(loss_history[-5:])
        else:
            loss_trend = current_loss
            loss_variance = 0.0
        
        # Gradient statistics
        gradient_norm = gradient_stats.get('norm', 0.0)
        gradient_variance = gradient_stats.get('variance', 0.0)
        
        # Training progress
        iteration = training_info.get('iteration', 0)
        epoch = training_info.get('epoch', 0)
        
        # Normalize values
        normalized_loss = min(current_loss / 10.0, 1.0)  # Cap at 1.0
        normalized_trend = min(loss_trend / 10.0, 1.0)
        normalized_grad_norm = min(gradient_norm / 10.0, 1.0)
        normalized_iteration = min(iteration / 1000.0, 1.0)
        
        state = np.array([
            normalized_loss,
            normalized_trend,
            loss_variance,
            normalized_grad_norm,
            gradient_variance,
            normalized_iteration
        ], dtype=np.float32)
        
        return torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state tensor
            training: Whether in training mode
        
        Returns:
            int: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Exploitation: best action according to Q-network
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=None):
        """
        Train the Q-network using experience replay
        
        Args:
            batch_size: Size of training batch
        """
        batch_size = batch_size or RL_CONFIG['batch_size']
        
        if len(self.memory) < batch_size:
            return 0.0  # Not enough experience yet
        
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.cat(next_states).to(DEVICE)
        dones = torch.BoolTensor(dones).to(DEVICE)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 
            SAFETY_CONFIG['gradient_clip_value']
        )
        
        self.optimizer.step()
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def action_to_params(self, action):
        """
        Convert action to loss adaptation parameters
        
        Args:
            action: Integer action
        
        Returns:
            dict: Loss adaptation parameters
        """
        if action not in self.action_mapping:
            action = 0  # Default to conservative
        
        params = self.action_mapping[action].copy()
        
        # Apply safety constraints
        params['main_weight'] = np.clip(
            params['main_weight'],
            SAFETY_CONFIG['min_loss_multiplier'],
            SAFETY_CONFIG['max_loss_multiplier']
        )
        
        return params
    
    def compute_reward(self, current_metrics, previous_metrics, adaptation_params):
        """
        Compute reward based on training improvement
        
        Args:
            current_metrics: Current training metrics
            previous_metrics: Previous training metrics
            adaptation_params: Applied adaptation parameters
        
        Returns:
            float: Computed reward
        """
        reward = 0.0
        
        # 1. Loss improvement
        prev = previous_metrics.get('loss', None)
        curr = current_metrics.get('loss', None)
        if prev is not None and prev > 0 and curr is not None:
            loss_improvement = (prev - curr) / prev
        else:
            loss_improvement = 0.0
        reward += loss_improvement * 10.0

        # 2. Gradient penalty
        if 'gradient_norm' in current_metrics:
            gradient_norm = current_metrics['gradient_norm']
            gradient_penalty = -abs(gradient_norm - 1.0) * 0.1
            reward += gradient_penalty

        # 3. Stability reward
        if current_metrics.get('stable', False):
            reward += 0.5
        else:
            reward -= 0.5

        # 4. Penalty for extreme adaptation
        mw = adaptation_params.get('main_weight', 1.0)
        reward -= abs(mw - 1.0) * 0.1

        return float(max(min(reward, 10.0), -10.0))
    
    def save_agent(self, filepath):
        """Save agent state to file"""
        state = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'total_rewards': self.total_rewards,
            'loss_history': self.loss_history,
            'memory': list(self.memory)  # Convert deque to list for pickling
        }
        
        torch.save(state, filepath)
        print(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            print(f"No saved agent found at {filepath}")
            return
        
        state = torch.load(filepath, map_location=DEVICE)
        
        self.q_network.load_state_dict(state['q_network_state_dict'])
        self.target_network.load_state_dict(state['target_network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epsilon = state['epsilon']
        self.total_steps = state['total_steps']
        self.total_rewards = state['total_rewards']
        self.loss_history = state['loss_history']
        
        # Restore memory
        self.memory = deque(state['memory'], maxlen=self.memory_size)
        
        print(f"Agent loaded from {filepath}")
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'average_reward': np.mean(self.total_rewards) if self.total_rewards else 0.0,
            'memory_size': len(self.memory),
            'q_loss_average': np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        }

if __name__ == "__main__":
    # Test the loss monitor agent
    print("Testing Loss Monitor Agent...")
    
    agent = LossMonitorAgent()
    
    # Test state creation
    loss_history = [1.0, 0.9, 0.8, 0.7, 0.6]
    gradient_stats = {'norm': 0.5, 'variance': 0.1}
    training_info = {'iteration': 100, 'epoch': 5}
    
    state = agent.get_state(loss_history, gradient_stats, training_info)
    print(f"State shape: {state.shape}")
    print(f"State values: {state}")
    
    # Test action selection
    action = agent.act(state)
    print(f"Selected action: {action}")
    
    # Test action to parameters conversion
    params = agent.action_to_params(action)
    print(f"Action parameters: {params}")
    
    # Test reward computation
    current_metrics = {'loss': 0.5, 'gradient_norm': 0.8, 'stable': True}
    previous_metrics = {'loss': 0.6, 'gradient_norm': 0.7, 'stable': True}
    reward = agent.compute_reward(current_metrics, previous_metrics, params)
    print(f"Computed reward: {reward}")
    
    print("Loss Monitor Agent test completed successfully!")
