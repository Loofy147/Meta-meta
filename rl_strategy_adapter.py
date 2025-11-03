"""
Reinforcement Learning Strategy Adapter

Implements a Deep Q-Network (DQN) agent that learns optimal strategy weights
and risk parameters dynamically based on market regime and performance feedback.

This goes beyond static backtesting by continuously adapting to changing
market conditions using online learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import psycopg2
import os
from dotenv import load_dotenv
import json

load_dotenv()


class MarketStateEncoder:
    """
    Encodes current market state into a fixed-size feature vector
    suitable for RL agent consumption.
    """
    
    def __init__(self):
        self.feature_dim = 32  # Size of state representation
        
    def encode(self, market_data: Dict) -> np.ndarray:
        """
        Converts raw market data into normalized state vector.
        
        Features include:
        - Recent price momentum (multiple timeframes)
        - Volatility regime
        - Volume profile
        - Strategy performance history
        - Market microstructure indicators
        """
        features = []
        
        # Price momentum features (5 timeframes)
        for tf in ['1m', '5m', '15m', '1h', '4h']:
            momentum = market_data.get(f'momentum_{tf}', 0.0)
            features.append(np.tanh(momentum))  # Normalize to [-1, 1]
        
        # Volatility (normalized by historical percentile)
        volatility = market_data.get('volatility', 0.0)
        features.append(min(volatility / 0.05, 1.0))  # Cap at 5%
        
        # Volume indicators
        volume_ratio = market_data.get('volume_ratio', 1.0)
        features.append(np.log1p(volume_ratio) / 3.0)  # Log-normalize
        
        # Recent strategy performance
        for strategy in ['rsi', 'macd', 'ml', 'sentiment', 'orderbook']:
            hit_rate = market_data.get(f'{strategy}_hit_rate', 0.5)
            features.append(hit_rate)
        
        # Sharpe ratio (clipped)
        sharpe = market_data.get('sharpe_ratio', 0.0)
        features.append(np.tanh(sharpe / 2.0))
        
        # Drawdown indicator
        drawdown = market_data.get('drawdown', 0.0)
        features.append(-drawdown)  # Negative = bad
        
        # Market regime indicators
        features.extend([
            market_data.get('trend_strength', 0.0),
            market_data.get('mean_reversion_score', 0.0),
            market_data.get('correlation_spy', 0.0),
        ])
        
        # Pad or truncate to fixed dimension
        features = features[:self.feature_dim]
        features.extend([0.0] * (self.feature_dim - len(features)))
        
        return np.array(features, dtype=np.float32)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for learning optimal action-value function.
    
    Actions represent discrete strategy configurations:
    - Aggressive (high confidence threshold, concentrated positions)
    - Moderate (balanced approach)
    - Conservative (low risk, diversified)
    - Defensive (reduce exposure, tighter stops)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class RLStrategyOptimizer:
    """
    Reinforcement learning agent that optimizes strategy parameters
    in real-time based on performance feedback.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 4,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 10000,
        memory_size: int = 10000,
        batch_size: int = 64
    ):
        """
        Args:
            state_dim: Dimension of market state vector
            action_dim: Number of discrete strategy configurations
            learning_rate: Learning rate for Q-network
            gamma: Discount factor for future rewards
            epsilon_start/end/decay: Exploration schedule parameters
            memory_size: Size of replay buffer
            batch_size: Mini-batch size for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q-networks (current and target for stability)
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Market state encoder
        self.encoder = MarketStateEncoder()
        
        # Step counter for epsilon decay
        self.steps = 0
        
        # Action definitions (strategy configs)
        self.action_configs = {
            0: {  # Aggressive
                'confidence_threshold': 0.45,
                'max_position_size_pct': 0.15,
                'max_portfolio_exposure_pct': 0.7,
                'strategy_weights': {'rsi': 0.15, 'macd': 0.15, 'ml': 0.3, 'sentiment': 0.2, 'orderbook': 0.2}
            },
            1: {  # Moderate
                'confidence_threshold': 0.55,
                'max_position_size_pct': 0.10,
                'max_portfolio_exposure_pct': 0.5,
                'strategy_weights': {'rsi': 0.2, 'macd': 0.2, 'ml': 0.25, 'sentiment': 0.15, 'orderbook': 0.2}
            },
            2: {  # Conservative
                'confidence_threshold': 0.65,
                'max_position_size_pct': 0.05,
                'max_portfolio_exposure_pct': 0.3,
                'strategy_weights': {'rsi': 0.25, 'macd': 0.25, 'ml': 0.2, 'sentiment': 0.1, 'orderbook': 0.2}
            },
            3: {  # Defensive
                'confidence_threshold': 0.75,
                'max_position_size_pct': 0.02,
                'max_portfolio_exposure_pct': 0.15,
                'strategy_weights': {'rsi': 0.3, 'macd': 0.3, 'ml': 0.15, 'sentiment': 0.05, 'orderbook': 0.2}
            }
        }
    
    def select_action(self, state: np.ndarray, exploit: bool = False) -> int:
        """
        Selects action using epsilon-greedy policy.
        
        Args:
            state: Current market state vector
            exploit: If True, always exploit (no exploration)
            
        Returns:
            Action index (strategy configuration)
        """
        # Decay epsilon
        if not exploit:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.steps / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
            )
            self.steps += 1
        
        # Exploration
        if not exploit and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Stores transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def calculate_reward(self, performance_metrics: Dict) -> float:
        """
        Calculates reward signal from trading performance.
        
        Reward function considers:
        - Profit/loss (primary)
        - Risk-adjusted returns (Sharpe ratio)
        - Drawdown penalty
        - Hit rate bonus
        """
        pnl = performance_metrics.get('pnl', 0.0)
        sharpe = performance_metrics.get('sharpe_ratio', 0.0)
        drawdown = performance_metrics.get('drawdown', 0.0)
        hit_rate = performance_metrics.get('hit_rate', 0.5)
        
        # Multi-component reward
        reward = 0.0
        
        # PnL component (scaled to reasonable range)
        reward += np.tanh(pnl / 1000.0) * 1.0
        
        # Risk-adjusted return bonus
        reward += min(sharpe / 2.0, 1.0) * 0.5
        
        # Hit rate bonus (above 50% baseline)
        reward += (hit_rate - 0.5) * 2.0 * 0.3
        
        # Drawdown penalty (squared to penalize large drawdowns heavily)
        reward -= (drawdown ** 2) * 2.0
        
        return reward
    
    def train_step(self):
        """
        Performs one training step using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Use current network to select best action
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate that action
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_strategy_config(self, action: int) -> Dict:
        """Returns strategy configuration for given action"""
        return self.action_configs[action]
    
    def save_model(self, path: str = "rl_models/dqn_strategy.pt"):
        """Saves trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "rl_models/dqn_strategy.pt"):
        """Loads trained model"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps = checkpoint['steps']
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")


def integrate_rl_optimizer():
    """
    Integration point for the RL optimizer into the main system.
    
    This should be called periodically (e.g., every hour) to:
    1. Encode current market state
    2. Get optimal action from RL agent
    3. Update system configuration
    4. Calculate reward from recent performance
    5. Train the agent
    """
    optimizer = RLStrategyOptimizer()
    
    # In production, this would:
    # 1. Fetch current market metrics from database
    # 2. Encode into state vector
    # 3. Select action and apply configuration
    # 4. Track performance and compute rewards
    # 5. Periodically train the agent
    
    print("RL Strategy Optimizer initialized and ready for integration.")
    return optimizer


if __name__ == '__main__':
    print("=== Reinforcement Learning Strategy Optimizer ===\n")
    
    # Initialize optimizer
    rl_optimizer = RLStrategyOptimizer()
    
    # Simulate a training episode
    print("Simulating training episode...\n")
    
    # Mock market state
    mock_market_data = {
        'momentum_1m': 0.02,
        'momentum_5m': 0.01,
        'momentum_15m': -0.005,
        'momentum_1h': 0.03,
        'momentum_4h': 0.015,
        'volatility': 0.025,
        'volume_ratio': 1.2,
        'rsi_hit_rate': 0.55,
        'macd_hit_rate': 0.52,
        'ml_hit_rate': 0.58,
        'sentiment_hit_rate': 0.48,
        'orderbook_hit_rate': 0.60,
        'sharpe_ratio': 1.5,
        'drawdown': 0.05,
        'trend_strength': 0.7,
        'mean_reversion_score': 0.3,
        'correlation_spy': 0.6
    }
    
    # Encode state
    state = rl_optimizer.encoder.encode(mock_market_data)
    print(f"Encoded state vector (first 10 dims): {state[:10]}")
    
    # Select action
    action = rl_optimizer.select_action(state)
    config = rl_optimizer.get_strategy_config(action)
    print(f"\nSelected action: {action}")
    print(f"Strategy configuration:")
    print(json.dumps(config, indent=2))
    
    # Mock reward calculation
    performance = {
        'pnl': 500.0,
        'sharpe_ratio': 1.8,
        'drawdown': 0.03,
        'hit_rate': 0.58
    }
    reward = rl_optimizer.calculate_reward(performance)
    print(f"\nCalculated reward: {reward:.3f}")
    
    # Store transition
    next_state = state + np.random.randn(rl_optimizer.state_dim) * 0.1
    rl_optimizer.store_transition(state, action, reward, next_state, done=False)
    
    # Training step (once enough experience is collected)
    print("\nRL optimizer ready for online learning.")
