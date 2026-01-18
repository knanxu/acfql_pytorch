import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union


class Dataset:
    """
    Offline RL Dataset for D4RL, Robomimic, and OGBench.
    
    Supports:
    - Random batch sampling
    - Sequential trajectory sampling
    - Frame stacking for visual observations
    - Image augmentation
    - Episode boundary handling
    """
    
    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initialize dataset from dictionary of numpy arrays.
        
        Args:
            data: Dictionary with keys like 'observations', 'actions',
                  'rewards', 'terminals'/'dones', etc.
        
        Note:
            Images should be stored in NumPy format (H, W, C).
            Use pytorch_format=True in sampling methods to convert to (C, H, W).
        """
        self._data = data
        self.size = len(next(iter(data.values())))
        
        # Optional features (can be set after initialization)
        self.frame_stack = None  # Number of frames to stack
        self.p_aug = None        # Image augmentation probability
        self.aug_padding = 4     # Padding for random crop augmentation
        self.return_next_actions = False
        self.pytorch_format = False  # Whether to convert images to PyTorch format (C, H, W)
        
        # Compute episode boundaries
        self._compute_episode_boundaries()
    
    def _compute_episode_boundaries(self):
        """Identify where episodes start and end in the dataset."""
        # Look for 'terminals' (D4RL) or 'dones' (common alternative)
        terminal_key = 'terminals' if 'terminals' in self._data else 'dones'
        
        if terminal_key in self._data:
            self.terminal_indices = np.where(self._data[terminal_key] > 0)[0]
            # Episode starts: beginning of dataset + after each terminal
            self.episode_starts = np.concatenate([[0], self.terminal_indices[:-1] + 1])
        else:
            # No terminal info: treat entire dataset as one episode
            self.terminal_indices = np.array([self.size - 1])
            self.episode_starts = np.array([0])
    
    @classmethod
    def create(cls, **fields):
        """
        Create dataset from keyword arguments.
        
        Example:
            dataset = Dataset.create(
                observations=obs_array,
                actions=act_array,
                rewards=rew_array,
                terminals=term_array
            )
        """
        data = {k: np.asarray(v) for k, v in fields.items()}
        return cls(data)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, key: str):
        """Access dataset fields like a dictionary."""
        return self._data[key]
    
    def __contains__(self, key: str):
        return key in self._data
    
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def copy(self, add_or_replace: Dict[str, np.ndarray] = None, **updates):
        """
        Create a copy of the dataset, optionally updating fields.
        
        Args:
            add_or_replace: Dictionary of fields to add or replace (JAX-style API)
            **updates: Fields to add or replace (alternative API)
        """
        new_data = self._data.copy()
        # Support both JAX-style API (add_or_replace=) and direct kwargs
        if add_or_replace is not None:
            new_data.update({k: np.asarray(v) for k, v in add_or_replace.items()})
        new_data.update({k: np.asarray(v) for k, v in updates.items()})
        
        new_dataset = Dataset(new_data)
        new_dataset.frame_stack = self.frame_stack
        new_dataset.p_aug = self.p_aug
        new_dataset.aug_padding = self.aug_padding
        new_dataset.return_next_actions = self.return_next_actions
        new_dataset.pytorch_format = self.pytorch_format
        return new_dataset
    
    def _apply_to_nested(self, func, data):
        """Apply function recursively to nested dict/list structures."""
        if isinstance(data, dict):
            return {k: self._apply_to_nested(func, v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._apply_to_nested(func, item) for item in data)
        else:
            return func(data)
    
    def _to_pytorch_format(self, data: np.ndarray) -> np.ndarray:
        """
        Convert image data from NumPy format to PyTorch format.
        
        Converts:
            - (H, W, C) -> (C, H, W) for single images
            - (B, H, W, C) -> (B, C, H, W) for batches
            - (B, T, H, W, C) -> (B, T, C, H, W) for sequences
        
        Args:
            data: Image data in NumPy format
        
        Returns:
            Image data in PyTorch format (channels first)
        """
        if data.ndim == 3:  # Single image: (H, W, C) -> (C, H, W)
            return data.transpose(2, 0, 1)
        elif data.ndim == 4:  # Batch: (B, H, W, C) -> (B, C, H, W)
            return data.transpose(0, 3, 1, 2)
        elif data.ndim == 5:  # Sequence: (B, T, H, W, C) -> (B, T, C, H, W)
            return data.transpose(0, 1, 4, 2, 3)
        else:
            return data
    
    def _is_image_data(self, data: np.ndarray) -> bool:
        """
        Check if data appears to be image data.
        
        Heuristic: Has 3+ dimensions and last dimension is 1, 3, or 4 (channels).
        """
        if data.ndim < 3:
            return False
        channels = data.shape[-1]
        return channels in [1, 3, 4]
    
    def _random_crop(self, img: np.ndarray, padding: int) -> np.ndarray:
        """Apply random crop augmentation to a single image."""
        h, w = img.shape[:2]
        padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
        
        crop_y = np.random.randint(0, 2 * padding + 1)
        crop_x = np.random.randint(0, 2 * padding + 1)
        
        return padded[crop_y:crop_y + h, crop_x:crop_x + w]
    
    def _augment_batch(self, batch: Dict[str, np.ndarray], keys: List[str]):
        """Apply image augmentation to specified keys in the batch."""
        for key in keys:
            if key not in batch:
                continue
            
            data = batch[key]
            # Only augment 4D image data (batch, height, width, channels)
            if data.ndim == 4:
                batch[key] = np.array([
                    self._random_crop(img, self.aug_padding) for img in data
                ])
    
    def sample(self, batch_size: int, indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            indices: Optional specific indices to sample (for reproducibility)
        
        Returns:
            Dictionary with sampled transitions.
            If pytorch_format=True, images are in (B, C, H, W) format.
            Otherwise, images are in (B, H, W, C) format (NumPy default).
        """
        if indices is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        
        # Basic sampling
        batch = self._apply_to_nested(lambda arr: arr[indices], self._data)
        
        # Add next actions if requested
        if self.return_next_actions:
            next_indices = np.minimum(indices + 1, self.size - 1)
            batch['next_actions'] = self._data['actions'][next_indices]
        
        # Handle frame stacking
        if self.frame_stack is not None:
            batch = self._stack_frames(batch, indices)
        
        # Apply augmentation (always done in NumPy format)
        if self.p_aug is not None and np.random.rand() < self.p_aug:
            self._augment_batch(batch, ['observations', 'next_observations'])
        
        # Convert to PyTorch format if requested
        if self.pytorch_format:
            for key in ['observations', 'next_observations']:
                if key in batch and self._is_image_data(batch[key]):
                    batch[key] = self._to_pytorch_format(batch[key])
        
        return batch
    
    def _stack_frames(self, batch: Dict, indices: np.ndarray) -> Dict:
        """Stack historical frames for current observations."""
        # Find episode start for each sampled index
        episode_start_indices = self.episode_starts[
            np.searchsorted(self.episode_starts, indices, side='right') - 1
        ]
        
        obs_frames = []
        next_obs_frames = []
        
        # Collect frames from t-k to t
        for k in range(self.frame_stack - 1, -1, -1):
            # Clamp to episode boundaries
            frame_indices = np.maximum(indices - k, episode_start_indices)
            
            obs_frame = self._apply_to_nested(
                lambda arr: arr[frame_indices], 
                self._data['observations']
            )
            obs_frames.append(obs_frame)
            
            # For next_obs, shift by one timestep
            if k > 0:
                next_obs_frames.append(obs_frame)
        
        # Add the actual next observation
        next_obs_frames.append(
            self._apply_to_nested(
                lambda arr: arr[indices],
                self._data['next_observations']
            )
        )
        
        # Concatenate along the last axis (channel dimension)
        batch['observations'] = self._apply_to_nested(
            lambda *frames: np.concatenate(frames, axis=-1),
            *obs_frames
        )
        batch['next_observations'] = self._apply_to_nested(
            lambda *frames: np.concatenate(frames, axis=-1),
            *next_obs_frames
        )
        
        return batch
    
    def sample_sequence(
        self, 
        batch_size: int, 
        sequence_length: int,
        discount: float = 0.99
    ) -> Dict[str, np.ndarray]:
        """
        Sample sequences of consecutive transitions.
        
        Useful for Transformer/RNN-based methods that need temporal context.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            discount: Discount factor for computing cumulative rewards
        
        Returns:
            Dictionary containing:
                - observations: Starting observation
                  Shape: (B, obs_dim) or (B, H, W, C) or (B, C, H, W) if pytorch_format=True
                - full_observations: All observations
                  Shape: (B, T, obs_dim) or (B, T, H, W, C) or (B, T, C, H, W) if pytorch_format=True
                - next_observations: Next observations
                  Shape: (B, T, obs_dim) or (B, T, H, W, C) or (B, T, C, H, W) if pytorch_format=True
                - actions: All actions (B, T, action_dim)
                - next_actions: Next step actions (B, T, action_dim)
                - rewards: Cumulative discounted rewards (B, T)
                - masks: Validity mask (B, T)
                - terminals: Terminal flags (B, T)
                - valid: Within-episode validity (B, T)
            
            where B=batch_size, T=sequence_length
        """
        # Sample valid starting indices
        max_start = self.size - sequence_length
        if max_start <= 0:
            raise ValueError(f"Dataset size {self.size} too small for sequence length {sequence_length}")
        
        start_indices = np.random.randint(0, max_start, size=batch_size)
        
        # Create sequence indices: (batch_size, sequence_length)
        sequence_indices = start_indices[:, None] + np.arange(sequence_length)[None, :]
        flat_indices = sequence_indices.ravel()
        
        # Helper to fetch and reshape data
        def fetch_sequence(key: str) -> np.ndarray:
            data = self._data[key][flat_indices]
            new_shape = (batch_size, sequence_length) + data.shape[1:]
            return data.reshape(new_shape)
        
        # Fetch all sequences
        obs_seq = fetch_sequence('observations')
        next_obs_seq = fetch_sequence('next_observations')
        action_seq = fetch_sequence('actions')
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
        
        # Compute next actions (look one step ahead)
        next_action_indices = np.minimum(sequence_indices + 1, self.size - 1).ravel()
        next_actions = self._data['actions'][next_action_indices].reshape(
            batch_size, sequence_length, -1
        )
        
        # Compute cumulative rewards and propagate episode information
        cumulative_rewards = np.zeros((batch_size, sequence_length))
        cumulative_masks = np.ones((batch_size, sequence_length))
        cumulative_terminals = np.zeros((batch_size, sequence_length))
        validity = np.ones((batch_size, sequence_length))
        
        # Initialize first timestep
        cumulative_rewards[:, 0] = reward_seq[:, 0].squeeze()
        cumulative_masks[:, 0] = mask_seq[:, 0].squeeze()
        cumulative_terminals[:, 0] = terminal_seq[:, 0].squeeze()
        
        # Discount powers for efficient computation
        discount_power = discount ** np.arange(sequence_length)
        
        # Propagate through sequence
        for t in range(1, sequence_length):
            # Accumulate discounted rewards
            cumulative_rewards[:, t] = (
                cumulative_rewards[:, t-1] + 
                reward_seq[:, t].squeeze() * discount_power[t]
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
        
        # Convert to PyTorch format if requested
        # Visual observations: (B, T, H, W, C) -> (B, T, C, H, W)
        if self.pytorch_format:
            if self._is_image_data(obs_seq):
                obs_seq = self._to_pytorch_format(obs_seq)
                next_obs_seq = self._to_pytorch_format(next_obs_seq)
            
            # Also convert initial observations
            initial_obs = self._data['observations'][start_indices]
            if self._is_image_data(initial_obs):
                initial_obs = self._to_pytorch_format(initial_obs)
        else:
            initial_obs = self._data['observations'][start_indices]
        
        return {
            'observations': initial_obs,  # Initial obs
            'full_observations': obs_seq,
            'next_observations': next_obs_seq,
            'actions': action_seq,
            'next_actions': next_actions,
            'rewards': cumulative_rewards,
            'masks': cumulative_masks,
            'terminals': cumulative_terminals,
            'valid': validity,
        }
    
    def to_replay_memory(self, memory, max_samples: Optional[int] = None):
        """
        Load dataset into a ReplayMemory object.
        
        Args:
            memory: ReplayMemory instance
            max_samples: Maximum number of samples to load (None = all)
        """
        n_samples = min(self.size, max_samples) if max_samples else self.size
        print(f"Loading {n_samples} transitions into ReplayMemory...")
        
        obs = self._data['observations']
        actions = self._data['actions']
        rewards = self._data['rewards']
        next_obs = self._data.get('next_observations', np.roll(obs, -1, axis=0))
        
        # Compute not_done from masks or terminals
        if 'masks' in self._data:
            not_done = self._data['masks']
        elif 'terminals' in self._data:
            not_done = 1.0 - self._data['terminals']
        elif 'dones' in self._data:
            not_done = 1.0 - self._data['dones']
        else:
            not_done = np.ones_like(rewards)
        
        for i in range(n_samples):
            memory.append(obs[i], actions[i], rewards[i], next_obs[i], not_done[i])
        
        print(f"Successfully loaded {n_samples} transitions.")
    
    def to_tensors(
        self, 
        batch_size: int, 
        device: Optional[torch.device] = None
    ) -> tuple:
        """
        Sample a batch and return as PyTorch tensors.
        
        Args:
            batch_size: Number of samples
            device: Target device for tensors
        
        Returns:
            Tuple of (states, actions, next_states, rewards, not_done)
        """
        batch = self.sample(batch_size)
        
        states = torch.from_numpy(batch['observations']).float()
        actions = torch.from_numpy(batch['actions']).float()
        next_states = torch.from_numpy(batch['next_observations']).float()
        rewards = torch.from_numpy(batch['rewards']).float()
        
        # Compute not_done
        if 'masks' in batch:
            not_done = torch.from_numpy(batch['masks']).float()
        elif 'terminals' in batch:
            not_done = 1.0 - torch.from_numpy(batch['terminals']).float()
        else:
            not_done = torch.ones_like(rewards)
        
        if device:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)
            not_done = not_done.to(device)
        
        return states, actions, next_states, rewards, not_done


def add_history(dataset: Dataset, history_length: int) -> Dataset:
    """
    Add observation and action history to dataset.
    
    Creates new fields 'observation_history' and 'action_history' containing
    the last `history_length - 1` observations/actions before each timestep.
    
    Args:
        dataset: Source dataset
        history_length: Number of historical steps to include
    
    Returns:
        New dataset with history fields added
    """
    if 'terminals' not in dataset and 'dones' not in dataset:
        raise ValueError("Dataset must have 'terminals' or 'dones' for history computation")
    
    size = dataset.size
    terminal_key = 'terminals' if 'terminals' in dataset else 'dones'
    terminals = dataset[terminal_key]
    
    terminal_locs = np.where(terminals > 0)[0]
    episode_starts = np.concatenate([[0], terminal_locs[:-1] + 1])
    
    indices = np.arange(size)
    episode_start_for_index = episode_starts[
        np.searchsorted(episode_starts, indices, side='right') - 1
    ]
    
    obs_history = []
    action_history = []
    
    # Collect history for each timestep
    for lag in range(1, history_length):
        # Get indices for t - lag, clamped to episode start
        hist_indices = np.maximum(indices - lag, episode_start_for_index)
        
        # Check which indices are outside the episode boundary
        is_outside = (indices - lag) < episode_start_for_index
        
        # Get observations and actions at historical indices
        hist_obs = dataset['observations'][hist_indices]
        hist_actions = dataset['actions'][hist_indices]
        
        # Zero out data that crosses episode boundaries
        if hist_obs.ndim > 1:
            hist_obs = hist_obs * (~is_outside[:, None])
        else:
            hist_obs = hist_obs * (~is_outside)
        
        if hist_actions.ndim > 1:
            hist_actions = hist_actions * (~is_outside[:, None])
        else:
            hist_actions = hist_actions * (~is_outside)
        
        obs_history.append(hist_obs)
        action_history.append(hist_actions)
    
    # Stack along new axis: (size, history_length - 1, feature_dim)
    obs_history = np.stack(obs_history, axis=1)
    action_history = np.stack(action_history, axis=1)
    
    return dataset.copy(
        observation_history=obs_history,
        action_history=action_history
    )
