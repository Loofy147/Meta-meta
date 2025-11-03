"""Implements a Reinforcement Learning agent for dynamic strategy optimization.

This module provides a Deep Q-Network (DQN) agent that learns to dynamically
select the best trading strategy configuration based on the current market state.
Instead of using fixed parameters, this RL agent can adapt to changing market
conditions (e.g., high volatility, strong trend) by choosing a pre-defined
"action" that corresponds to a specific set of strategy weights and risk
parameters.

The key components are:
- **MarketStateEncoder**: Compresses a wide range of market data (momentum,
  volatility, performance metrics) into a fixed-size vector representation
  (the "state") for the RL agent.
- **DQNNetwork**: A neural network that learns to predict the expected future
  reward (Q-value) for taking each action in a given state.
- **RLStrategyOptimizer**: The main agent class that manages the learning
  process, including the epsilon-greedy exploration strategy, experience replay,
  the reward function, and the training loop.
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
    """Encodes raw market data into a fixed-size state vector for the RL agent.

    This class is responsible for feature engineering. It takes a dictionary of
    various market metrics and transforms them into a normalized, numerical
    numpy array that can be fed into the neural network.
    """

    def __init__(self, feature_dim: int = 32):
        """Initializes the MarketStateEncoder.

        Args:
            feature_dim: The fixed dimensionality of the output state vector.
        """
        self.feature_dim = feature_dim

    def encode(self, market_data: Dict) -> np.ndarray:
        """Converts a dictionary of market data into a normalized state vector.

        The features include metrics like multi-timeframe momentum, volatility,
        volume profiles, recent strategy performance, Sharpe ratio, and drawdown.
        Features are normalized to a consistent range (e.g., [-1, 1] or [0, 1])
        to help with model training.

        Args:
            market_data: A dictionary containing various raw market metrics.

        Returns:
            A numpy array of shape (feature_dim,) representing the market state.
        """
        features = []

        # Normalize momentum features using tanh
        for tf in ['1m', '5m', '15m', '1h', '4h']:
            features.append(np.tanh(market_data.get(f'momentum_{tf}', 0.0)))

        # Normalize volatility and volume
        features.append(min(market_data.get('volatility', 0.0) / 0.05, 1.0))
        features.append(np.log1p(market_data.get('volume_ratio', 1.0)) / 3.0)

        # Add recent strategy performance (hit rates are already [0, 1])
        for strategy in ['rsi', 'macd', 'ml', 'sentiment', 'orderbook']:
            features.append(market_data.get(f'{strategy}_hit_rate', 0.5))

        # Add overall performance metrics
        features.append(np.tanh(market_data.get('sharpe_ratio', 0.0) / 2.0))
        features.append(-market_data.get('drawdown', 0.0))  # Drawdown is a negative signal

        # Add market regime indicators
        features.extend([
            market_data.get('trend_strength', 0.0),
            market_data.get('mean_reversion_score', 0.0)
        ])

        # Pad or truncate to ensure a fixed feature dimension
        padded_features = features[:self.feature_dim]
        padded_features.extend([0.0] * (self.feature_dim - len(padded_features)))

        return np.array(padded_features, dtype=np.float32)


class DQNNetwork(nn.Module):
    """A Deep Q-Network (DQN) to approximate the optimal action-value function.

    This neural network takes the market state as input and outputs the predicted
    Q-value for each possible action. The actions correspond to different pre-defined
    strategy configurations (e.g., 'aggressive', 'conservative').
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initializes the DQN architecture.

        Args:
            state_dim: The dimensionality of the input state vector.
            action_dim: The number of discrete actions the agent can take.
            hidden_dim: The size of the hidden layers.
        """
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network."""
        return self.network(state)


class RLStrategyOptimizer:
    """An RL agent that learns to optimize trading strategy configurations.

    This class implements the full logic for a Deep Q-Learning agent, including:
    - An epsilon-greedy policy for balancing exploration and exploitation.
    - An experience replay buffer to decorrelate experiences for stable training.
    - A reward function to translate trading performance into a learning signal.
    - A training loop that uses a target network for stability (Double DQN).
    """

    def __init__(self, state_dim: int = 32, action_dim: int = 4, **kwargs):
        """Initializes the RL agent and its components.

        Args:
            state_dim: The dimensionality of the state vector.
            action_dim: The number of discrete actions.
            **kwargs: Hyperparameters for the DQN agent, such as learning_rate,
                gamma, epsilon schedule, memory_size, and batch_size.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.1)
        self.epsilon_decay = kwargs.get('epsilon_decay', 10000)
        self.batch_size = kwargs.get('batch_size', 64)

        # Initialize the main and target Q-networks for stable learning.
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=kwargs.get('learning_rate', 1e-4))
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss for robustness
        self.memory = deque(maxlen=kwargs.get('memory_size', 10000))
        self.encoder = MarketStateEncoder(state_dim)
        self.steps = 0
        
        # Actions are mapped to predefined configurations for risk and strategy weights.
        self.action_configs = {
            0: {'name': 'Aggressive', 'confidence_threshold': 0.45, 'max_position_size_pct': 0.15, 'strategy_weights': {'ml': 0.4, 'orderbook': 0.3}},
            1: {'name': 'Moderate', 'confidence_threshold': 0.55, 'max_position_size_pct': 0.10, 'strategy_weights': {'ml': 0.3, 'rsi': 0.2, 'macd': 0.2}},
            2: {'name': 'Conservative', 'confidence_threshold': 0.65, 'max_position_size_pct': 0.05, 'strategy_weights': {'rsi': 0.3, 'macd': 0.3, 'ml': 0.2}},
            3: {'name': 'Defensive', 'confidence_threshold': 0.75, 'max_position_size_pct': 0.02, 'strategy_weights': {'rsi': 0.5, 'macd': 0.5}},
        }

    def select_action(self, state: np.ndarray, exploit: bool = False) -> int:
        """Selects an action using an epsilon-greedy policy.

        During training, the agent will "explore" by choosing a random action
        with probability epsilon, and "exploit" its current knowledge by
        choosing the best-known action otherwise. Epsilon decays over time.

        Args:
            state: The current market state vector.
            exploit: If True, forces the agent to always choose the best action.

        Returns:
            The integer index of the chosen action.
        """
        if not exploit:
            # Linearly decay epsilon from start to end over `epsilon_decay` steps.
            self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 / self.epsilon_decay))
            self.steps += 1
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)

        # Exploitation: choose the action with the highest Q-value.
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Stores a (state, action, reward, next_state, done) tuple in the experience replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def calculate_reward(self, performance_metrics: Dict) -> float:
        """Calculates a reward signal based on recent trading performance.

        The reward function is critical. This one is a composite signal that
        rewards profit but heavily penalizes risk (drawdown) and rewards
        risk-adjusted returns (Sharpe ratio).

        Args:
            performance_metrics: A dictionary with keys like 'pnl', 'sharpe_ratio',
                'drawdown', and 'hit_rate'.

        Returns:
            A single float value representing the reward.
        """
        pnl = performance_metrics.get('pnl', 0.0)
        sharpe = performance_metrics.get('sharpe_ratio', 0.0)
        drawdown = performance_metrics.get('drawdown', 0.0)
        hit_rate = performance_metrics.get('hit_rate', 0.5)

        # Primary reward from scaled PnL
        reward = np.tanh(pnl / 1000.0) * 1.0
        # Bonus for risk-adjusted returns
        reward += np.tanh(sharpe / 2.0) * 0.5
        # Bonus for accuracy
        reward += (hit_rate - 0.5) * 0.3
        # Heavy penalty for drawdowns
        reward -= (drawdown ** 2) * 2.0
        return reward

    def train_step(self) -> Optional[float]:
        """Performs one training step on a mini-batch from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get Q-values for the actions that were actually taken.
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Calculate target Q-values using the Double DQN algorithm for stability.
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, tau: float = 0.005):
        """Performs a soft update of the target network's weights."""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_strategy_config(self, action: int) -> Dict:
        """Returns the dictionary of parameters for a given action index."""
        return self.action_configs[action]

    def save_model(self, path: str = "rl_models/dqn_strategy.pt"):
        """Saves the trained model and optimizer state to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)
        print(f"RL model saved to {path}")

    def load_model(self, path: str = "rl_models/dqn_strategy.pt"):
        """Loads a pre-trained model and its state from a file."""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps = checkpoint.get('steps', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            print(f"RL model loaded from {path}")
        else:
            print(f"No RL model found at {path}, starting with a new model.")


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
