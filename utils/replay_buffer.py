"""
Replay Buffer for Offline-to-Online RL

Extends Dataset class to support:
- Creating buffer from example transition
- Creating buffer from initial offline dataset
- Adding new transitions during online training
- Clearing buffer
"""

import numpy as np
from typing import Dict, Any, Optional

from utils.datasets import Dataset


def get_size(data: Dict[str, np.ndarray]) -> int:
    """Get size (first dimension) of the data dictionary."""
    return len(next(iter(data.values())))


class ReplayBuffer(Dataset):
    """
    Replay buffer for offline-to-online reinforcement learning.
    
    Extends Dataset to support adding transitions during online training.
    Inherits all sampling methods from Dataset (sample, sample_sequence, etc.)
    """
    
    @classmethod
    def create(cls, transition: Dict[str, Any], size: int) -> 'ReplayBuffer':
        """
        Create a replay buffer from an example transition.
        
        Args:
            transition: Example transition dict with keys like 
                        'observations', 'actions', 'rewards', etc.
            size: Maximum size of the replay buffer.
        
        Returns:
            Empty replay buffer with pre-allocated arrays.
        
        Example:
            transition = {
                'observations': np.zeros(obs_dim),
                'actions': np.zeros(action_dim),
                'rewards': 0.0,
                'next_observations': np.zeros(obs_dim),
                'terminals': 0.0,
                'masks': 1.0,
            }
            buffer = ReplayBuffer.create(transition, size=1000000)
        """
        def create_buffer(example):
            example = np.asarray(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        
        buffer_dict = {k: create_buffer(v) for k, v in transition.items()}
        
        buffer = cls(buffer_dict)
        buffer.max_size = size
        buffer.size = 0
        buffer.pointer = 0
        return buffer
    
    @classmethod
    def create_from_initial_dataset(
        cls, 
        init_dataset: Dataset, 
        size: int,
        copy_data: bool = True
    ) -> 'ReplayBuffer':
        """
        Create a replay buffer initialized with an offline dataset.
        
        This is the main method for offline-to-online RL, where we start
        with an offline dataset and add online transitions.
        
        Args:
            init_dataset: Initial offline dataset (Dataset object or dict).
            size: Maximum size of the replay buffer.
            copy_data: If True, copy data from init_dataset. 
                      If False, reference the same arrays (faster but modifies original).
        
        Returns:
            Replay buffer initialized with the offline data.
        
        Example:
            # Load offline dataset
            train_dataset = Dataset.create(**offline_data)
            
            # Create buffer for online training (1M capacity)
            buffer = ReplayBuffer.create_from_initial_dataset(
                train_dataset, 
                size=1000000
            )
        """
        # Get data dict from Dataset or use directly if already dict
        if isinstance(init_dataset, Dataset):
            init_data = dict(init_dataset.items())
            init_size = len(init_dataset)
        else:
            init_data = init_dataset
            init_size = get_size(init_data)
        
        def create_buffer(init_buffer):
            init_buffer = np.asarray(init_buffer)
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            # Copy initial data
            copy_len = min(len(init_buffer), size)
            if copy_data:
                buffer[:copy_len] = init_buffer[:copy_len].copy()
            else:
                buffer[:copy_len] = init_buffer[:copy_len]
            return buffer
        
        buffer_dict = {k: create_buffer(v) for k, v in init_data.items()}
        
        buffer = cls(buffer_dict)
        buffer.max_size = size
        buffer.size = min(init_size, size)
        buffer.pointer = buffer.size % size
        
        # Copy dataset settings
        if isinstance(init_dataset, Dataset):
            buffer.frame_stack = init_dataset.frame_stack
            buffer.p_aug = init_dataset.p_aug
            buffer.aug_padding = init_dataset.aug_padding
            buffer.return_next_actions = init_dataset.return_next_actions
            buffer.pytorch_format = init_dataset.pytorch_format
        
        return buffer
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """Initialize replay buffer."""
        super().__init__(data)
        
        # Buffer-specific attributes
        self.max_size = get_size(self._data)
        self.size = 0
        self.pointer = 0
    
    def __len__(self):
        """Return current size of buffer (not max_size)."""
        return self.size
    
    def add_transition(self, transition: Dict[str, Any]):
        """
        Add a single transition to the replay buffer.
        
        Args:
            transition: Dict with same keys as buffer.
        
        Example:
            buffer.add_transition({
                'observations': obs,
                'actions': action,
                'rewards': reward,
                'next_observations': next_obs,
                'terminals': done,
                'masks': 1.0 - done,
            })
        """
        for key, value in transition.items():
            if key in self._data:
                self._data[key][self.pointer] = np.asarray(value)
        
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def add_transitions(self, transitions: Dict[str, np.ndarray]):
        """
        Add multiple transitions to the replay buffer.
        
        Args:
            transitions: Dict with arrays of transitions.
        
        Example:
            buffer.add_transitions({
                'observations': obs_batch,      # (N, obs_dim)
                'actions': action_batch,        # (N, action_dim)
                'rewards': reward_batch,        # (N,)
                'next_observations': next_obs_batch,
                'terminals': done_batch,
                'masks': mask_batch,
            })
        """
        batch_size = len(next(iter(transitions.values())))
        
        for i in range(batch_size):
            transition = {k: v[i] for k, v in transitions.items()}
            self.add_transition(transition)
    
    def clear(self):
        """Clear the replay buffer (reset size and pointer)."""
        self.size = 0
        self.pointer = 0
    
    def sample(self, batch_size: int, indices: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Overrides parent to use current size instead of total array size.
        
        Args:
            batch_size: Number of transitions to sample.
            indices: Optional specific indices to sample.
        
        Returns:
            Dictionary with sampled transitions.
        """
        if indices is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        
        # Use parent's sampling logic
        batch = self._apply_to_nested(lambda arr: arr[indices], self._data)
        
        # Handle frame stacking
        if self.frame_stack is not None:
            batch = self._stack_frames(batch, indices)
        
        # Apply augmentation
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self._augment_batch(batch, ['observations', 'next_observations'])
        
        # Convert to PyTorch format if requested
        if self.pytorch_format:
            for key in ['observations', 'next_observations']:
                if key in batch and self._is_image_data(batch[key]):
                    batch[key] = self._to_pytorch_format(batch[key])
        
        return batch
    
    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        discount: float = 0.99,
    ) -> Dict[str, np.ndarray]:
        """
        Sample sequences for action chunking.
        
        Overrides parent to use current size instead of total array size.
        
        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Length of each sequence.
            discount: Discount factor for computing discounted returns.
        
        Returns:
            Dictionary with sampled sequences including:
                - observations: Initial observations (B, obs_dim)
                - full_observations: All observations (B, T, obs_dim)
                - next_observations: Next observations (B, T, obs_dim)
                - actions: Actions (B, T, action_dim)
                - rewards: Cumulative discounted rewards (B, T)
                - masks: Validity masks (B, T)
                - terminals: Terminal flags (B, T)
                - valid: Within-episode validity (B, T)
        """
        # Sample starting indices (ensure we can get full sequences)
        max_start = max(1, self.size - sequence_length)
        start_indices = np.random.randint(0, max_start, size=batch_size)
        
        # Create sequence indices: (batch_size, sequence_length)
        sequence_indices = start_indices[:, None] + np.arange(sequence_length)[None, :]
        # Clip to valid range
        sequence_indices = np.clip(sequence_indices, 0, self.size - 1)
        flat_indices = sequence_indices.ravel()
        
        # Helper to fetch and reshape data
        def fetch_sequence(key: str) -> np.ndarray:
            data = self._data[key][flat_indices]
            new_shape = (batch_size, sequence_length) + data.shape[1:]
            return data.reshape(new_shape)
        
        # Fetch all sequences
        obs_seq = fetch_sequence('observations')
        action_seq = fetch_sequence('actions')
        
        # Handle next_observations (may not exist in all datasets)
        if 'next_observations' in self._data:
            next_obs_seq = fetch_sequence('next_observations')
        else:
            # Use shifted observations
            next_indices = np.clip(sequence_indices + 1, 0, self.size - 1)
            next_obs_seq = self._data['observations'][next_indices.ravel()].reshape(
                batch_size, sequence_length, -1
            )
        
        # Fetch rewards
        reward_seq = fetch_sequence('rewards')
        
        # Handle masks and terminals
        if 'masks' in self._data:
            mask_seq = fetch_sequence('masks')
        else:
            mask_seq = np.ones_like(reward_seq)
        
        terminal_key = 'terminals' if 'terminals' in self._data else 'dones'
        if terminal_key in self._data:
            terminal_seq = fetch_sequence(terminal_key)
        else:
            terminal_seq = np.zeros_like(reward_seq)
        
        # Compute cumulative rewards and propagate episode information
        cumulative_rewards = np.zeros((batch_size, sequence_length))
        cumulative_masks = np.ones((batch_size, sequence_length))
        cumulative_terminals = np.zeros((batch_size, sequence_length))
        validity = np.ones((batch_size, sequence_length))
        
        # Initialize first timestep
        cumulative_rewards[:, 0] = reward_seq[:, 0].squeeze()
        cumulative_masks[:, 0] = mask_seq[:, 0].squeeze()
        cumulative_terminals[:, 0] = terminal_seq[:, 0].squeeze()
        
        # Propagate through sequence
        for t in range(1, sequence_length):
            # Accumulate discounted rewards
            cumulative_rewards[:, t] = (
                cumulative_rewards[:, t-1] + 
                reward_seq[:, t].squeeze() * (discount ** t)
            )
            
            # Masks: take minimum (all must be valid)
            cumulative_masks[:, t] = np.minimum(
                cumulative_masks[:, t-1],
                mask_seq[:, t].squeeze()
            )
            
            # Terminals: take maximum (any terminal propagates)
            cumulative_terminals[:, t] = np.maximum(
                cumulative_terminals[:, t-1],
                terminal_seq[:, t].squeeze()
            )
            
            # Validity: 0 if previous timestep was terminal
            validity[:, t] = 1.0 - cumulative_terminals[:, t-1]
        
        # Initial observations
        initial_obs = self._data['observations'][start_indices]
        
        result = {
            'observations': initial_obs,              # (B, obs_dim)
            'full_observations': obs_seq,             # (B, T, obs_dim)
            'next_observations': next_obs_seq,        # (B, T, obs_dim)
            'actions': action_seq,                    # (B, T, action_dim)
            'rewards': cumulative_rewards,            # (B, T)
            'masks': cumulative_masks,                # (B, T)
            'terminals': cumulative_terminals,        # (B, T)
            'valid': validity,                        # (B, T)
        }
        
        # Convert to PyTorch format if requested
        if self.pytorch_format:
            for key in ['observations', 'full_observations', 'next_observations']:
                if key in result and self._is_image_data(result[key]):
                    result[key] = self._to_pytorch_format(result[key])
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'size': self.size,
            'max_size': self.max_size,
            'pointer': self.pointer,
            'utilization': self.size / self.max_size,
        }
    
    def __repr__(self):
        return f"ReplayBuffer(size={self.size}, max_size={self.max_size}, keys={list(self.keys())})"
