"""Lazy-loading image dataset for memory-efficient training with robomimic image data."""

import numpy as np
import h5py
from typing import Dict, List, Optional, Any
import torch


class LazyImageDataset:
    """
    Lazy-loading dataset for image-based robomimic data.

    Instead of loading all images into memory, this dataset keeps the HDF5 file
    open and loads images on-the-fly during sampling. This dramatically reduces
    memory usage for large image datasets.
    """

    def __init__(self, data: Dict[str, Any]):
        """Initialize lazy image dataset.

        Args:
            data: Dictionary containing:
                - dataset_path: Path to HDF5 file
                - shape_meta: Shape metadata dict
                - actions: Pre-loaded actions array
                - rewards: Pre-loaded rewards array
                - terminals: Pre-loaded terminals array
                - masks: Pre-loaded masks array
                - episode_starts: List of episode start indices
                - episode_lengths: List of episode lengths
                - image_size: Optional image resize target
        """
        self.dataset_path = data['dataset_path']
        self.shape_meta = data['shape_meta']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.terminals = data['terminals']
        self.masks = data['masks']
        self.episode_starts = data['episode_starts']
        self.episode_lengths = data['episode_lengths']
        self.image_size = data.get('image_size', None)

        self.size = len(self.actions)

        # Keep HDF5 file open for lazy loading
        self.h5_file = h5py.File(self.dataset_path, 'r')
        self.demos = list(self.h5_file["data"].keys())
        inds = np.argsort([int(elem.split("_")[-1]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        # Optional features
        self.pytorch_format = False
        self.return_next_actions = False

        # Compute episode boundaries
        self._compute_episode_boundaries()

    def _compute_episode_boundaries(self):
        """Identify where episodes start and end in the dataset."""
        self.terminal_indices = np.where(self.terminals > 0)[0]
        self.episode_starts_array = np.array([0] + [idx + 1 for idx in self.terminal_indices[:-1]])
        self.episode_ends = self.terminal_indices

    def _get_episode_and_local_idx(self, global_idx):
        """Convert global index to (episode_idx, local_idx) pair."""
        for ep_idx, (start, length) in enumerate(zip(self.episode_starts, self.episode_lengths)):
            if start <= global_idx < start + length:
                return ep_idx, global_idx - start
        raise ValueError(f"Index {global_idx} out of bounds")

    def _load_observation(self, global_idx):
        """Load observation at given index from HDF5 file."""
        ep_idx, local_idx = self._get_episode_and_local_idx(global_idx)
        demo_name = self.demos[ep_idx]

        obs_dict = {}
        for key in self.shape_meta["obs"].keys():
            dataset_key = key.replace("_image", "") if key.endswith("_image") else key

            if dataset_key in self.h5_file[f"data/{demo_name}/obs"]:
                obs_data = self.h5_file[f"data/{demo_name}/obs/{dataset_key}"][local_idx]

                # Process images
                if self.shape_meta["obs"][key]["type"] == "rgb":
                    obs_data = self._process_single_image(obs_data)

                obs_dict[key] = obs_data.astype(np.float32)
            else:
                # Create dummy data if key not found
                obs_shape = self.shape_meta["obs"][key]["shape"]
                obs_dict[key] = np.zeros(obs_shape, dtype=np.float32)

        return obs_dict

    def _load_next_observation(self, global_idx):
        """Load next observation at given index from HDF5 file."""
        ep_idx, local_idx = self._get_episode_and_local_idx(global_idx)
        demo_name = self.demos[ep_idx]

        obs_dict = {}
        for key in self.shape_meta["obs"].keys():
            dataset_key = key.replace("_image", "") if key.endswith("_image") else key

            if dataset_key in self.h5_file[f"data/{demo_name}/next_obs"]:
                obs_data = self.h5_file[f"data/{demo_name}/next_obs/{dataset_key}"][local_idx]

                # Process images
                if self.shape_meta["obs"][key]["type"] == "rgb":
                    obs_data = self._process_single_image(obs_data)

                obs_dict[key] = obs_data.astype(np.float32)
            else:
                # Create dummy data if key not found
                obs_shape = self.shape_meta["obs"][key]["shape"]
                obs_dict[key] = np.zeros(obs_shape, dtype=np.float32)

        return obs_dict

    def _process_single_image(self, image):
        """Process a single image: normalize and convert format."""
        original_shape = image.shape
        original_dtype = image.dtype

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to (C, H, W) if needed
        if image.ndim == 3:
            if image.shape[-1] in [1, 3]:  # (H, W, C)
                image = np.transpose(image, (2, 0, 1))
            elif image.shape[0] in [1, 3]:  # Already (C, H, W)
                pass  # Already in correct format
            else:
                # Ambiguous case - assume (H, W, C) if last dim is smallest
                if image.shape[-1] < image.shape[0]:
                    image = np.transpose(image, (2, 0, 1))

        # Resize if needed
        if self.image_size is not None:
            try:
                import cv2
                c, h, w = image.shape
                if (h, w) != (self.image_size, self.image_size):
                    resized = np.zeros((c, self.image_size, self.image_size), dtype=image.dtype)
                    for i in range(c):
                        resized[i] = cv2.resize(
                            image[i],
                            (self.image_size, self.image_size),
                            interpolation=cv2.INTER_AREA
                        )
                    image = resized
            except ImportError:
                pass  # Skip resize if cv2 not available

        return image

    def sample(self, batch_size: int, **kwargs) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        # Load observations lazily
        observations = {}
        next_observations = {}

        for idx in indices:
            obs = self._load_observation(idx)
            next_obs = self._load_next_observation(idx)

            for key in obs.keys():
                if key not in observations:
                    observations[key] = []
                    next_observations[key] = []
                observations[key].append(obs[key])
                next_observations[key].append(next_obs[key])

        # Stack observations
        for key in observations.keys():
            observations[key] = np.stack(observations[key], axis=0)
            next_observations[key] = np.stack(next_observations[key], axis=0)

        batch = {
            'observations': observations,
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'terminals': self.terminals[indices],
            'masks': self.masks[indices],
            'next_observations': next_observations,
        }

        return batch

    def sample_sequence(self, batch_size: int, sequence_length: int, **kwargs) -> Dict[str, np.ndarray]:
        """Sample sequences of transitions for action chunking."""
        # Sample starting indices that allow for full sequences
        max_start = self.size - sequence_length
        start_indices = np.random.randint(0, max_start + 1, size=batch_size)

        # Build sequences
        observations = {}
        actions_seq = []
        valid_mask = []

        for start_idx in start_indices:
            # Load observation sequence
            seq_obs = {}
            seq_actions = []
            seq_valid = []

            for offset in range(sequence_length):
                idx = start_idx + offset

                # Check if we're still in the same episode
                if idx < self.size and not (offset > 0 and self.terminals[idx - 1] > 0):
                    # Load observation for this timestep
                    obs = self._load_observation(idx)
                    for key in obs.keys():
                        if key not in seq_obs:
                            seq_obs[key] = []
                        seq_obs[key].append(obs[key])

                    seq_actions.append(self.actions[idx])
                    seq_valid.append(1.0)
                else:
                    # Pad with zeros if we cross episode boundary
                    # Load a dummy observation and zero it out
                    obs = self._load_observation(start_idx)  # Use first obs as template
                    for key in obs.keys():
                        if key not in seq_obs:
                            seq_obs[key] = []
                        seq_obs[key].append(np.zeros_like(obs[key]))

                    seq_actions.append(np.zeros_like(self.actions[0]))
                    seq_valid.append(0.0)

            # Stack sequence for this batch element
            for key in seq_obs.keys():
                if key not in observations:
                    observations[key] = []
                observations[key].append(np.stack(seq_obs[key], axis=0))

            actions_seq.append(np.stack(seq_actions, axis=0))
            valid_mask.append(np.array(seq_valid))

        # Stack batch
        for key in observations.keys():
            observations[key] = np.stack(observations[key], axis=0)

        batch = {
            'observations': observations,
            'actions': np.stack(actions_seq, axis=0),
            'valid': np.stack(valid_mask, axis=0),
        }

        return batch

    def copy(self, add_or_replace: Dict[str, np.ndarray] = None, **updates):
        """
        Create a copy of the dataset, optionally updating fields.

        Args:
            add_or_replace: Dictionary of fields to add or replace
            **updates: Fields to add or replace (alternative API)
        """
        # Create new data dict
        new_data = {
            'dataset_path': self.dataset_path,
            'shape_meta': self.shape_meta,
            'actions': self.actions.copy(),
            'rewards': self.rewards.copy(),
            'terminals': self.terminals.copy(),
            'masks': self.masks.copy(),
            'episode_starts': self.episode_starts.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'image_size': self.image_size,
        }

        # Apply updates
        if add_or_replace is not None:
            for k, v in add_or_replace.items():
                if k in new_data:
                    new_data[k] = np.asarray(v)
        for k, v in updates.items():
            if k in new_data:
                new_data[k] = np.asarray(v)

        # Create new dataset
        new_dataset = LazyImageDataset(new_data)
        new_dataset.pytorch_format = self.pytorch_format
        new_dataset.return_next_actions = self.return_next_actions

        return new_dataset

    def __getitem__(self, key):
        """Access dataset fields like a dictionary."""
        if key == 'actions':
            return self.actions
        elif key == 'rewards':
            return self.rewards
        elif key == 'terminals':
            return self.terminals
        elif key == 'masks':
            return self.masks
        elif key == 'observations':
            # This would load all observations - not recommended!
            raise NotImplementedError("Cannot access all observations at once in lazy dataset. Use sample() instead.")
        else:
            raise KeyError(f"Unknown key: {key}")

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    @classmethod
    def create(cls, **kwargs):
        """Create dataset from keyword arguments."""
        return cls(kwargs)
