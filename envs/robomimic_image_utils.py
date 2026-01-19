"""Robomimic image environment utilities.

Provides functions to create image-based robomimic environments and datasets.
Ported from much-ado-about-noising repository.
"""

import os
from os.path import expanduser
import collections

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from utils.datasets import Dataset


def get_shape_meta_from_dataset(dataset_path, obs_keys=None, use_eye_in_hand=True):
    """Extract shape metadata from robomimic dataset.

    Args:
        dataset_path: Path to the HDF5 dataset
        obs_keys: List of observation keys to use (if None, auto-detect from dataset)
        use_eye_in_hand: Whether to include eye-in-hand camera

    Returns:
        shape_meta: Dictionary containing observation and action shapes
    """
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        demo = demos[0]

        # Get action shape
        actions = f["data/{}/actions".format(demo)][()]
        action_shape = actions.shape[1:]

        # Get observation shapes
        obs_group = f["data/{}/obs".format(demo)]

        # Auto-detect observation keys if not provided
        if obs_keys is None:
            obs_keys = []
            # Add image observations
            for key in obs_group.keys():
                if "image" in key or key in ["agentview", "robot0_eye_in_hand"]:
                    if not use_eye_in_hand and "eye_in_hand" in key:
                        continue
                    obs_keys.append(key)
            # Add proprioceptive observations
            for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
                if key in obs_group:
                    obs_keys.append(key)

        # Build shape_meta
        shape_meta = {
            "action": {"shape": action_shape},
            "obs": {}
        }

        for key in obs_keys:
            if key not in obs_group:
                continue

            data = obs_group[key][()]
            obs_shape = data.shape[1:]

            # Determine observation type
            if "image" in key or key in ["agentview", "robot0_eye_in_hand"]:
                obs_type = "rgb"
                # Ensure key ends with _image for consistency
                if not key.endswith("_image"):
                    key_with_suffix = f"{key}_image"
                else:
                    key_with_suffix = key

                # Convert image shape from (H, W, C) to (C, H, W) for PyTorch
                if len(obs_shape) == 3 and obs_shape[-1] in [1, 3]:
                    obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (H, W, C) -> (C, H, W)
            else:
                obs_type = "low_dim"
                key_with_suffix = key

            shape_meta["obs"][key_with_suffix] = {
                "shape": obs_shape,
                "type": obs_type
            }

    return shape_meta


def make_env_and_dataset_image(env_name, seed=0, image_size=84, use_eye_in_hand=True):
    """Create image-based robomimic environment and dataset.

    Args:
        env_name: Environment name (e.g., "lift-mh-image")
        seed: Random seed
        image_size: Size to resize images to (if None, use original size)
        use_eye_in_hand: Whether to use eye-in-hand camera

    Returns:
        env: Training environment
        eval_env: Evaluation environment
        dataset: Training dataset
        shape_meta: Shape metadata dictionary
    """
    # Parse environment name
    if not env_name.endswith("-image"):
        raise ValueError(f"Environment name must end with '-image': {env_name}")

    base_name = env_name.replace("-image", "")
    task, dataset_type = base_name.split("-")[:2]

    # Get dataset path
    dataset_path = _check_dataset_exists_image(task, dataset_type)

    # Get shape metadata from dataset
    shape_meta = get_shape_meta_from_dataset(
        dataset_path,
        obs_keys=None,  # Auto-detect
        use_eye_in_hand=use_eye_in_hand
    )

    # Update image sizes if specified
    if image_size is not None:
        for key, value in shape_meta["obs"].items():
            if value["type"] == "rgb":
                # Assume images are (C, H, W) format
                c, h, w = value["shape"]
                shape_meta["obs"][key]["shape"] = (c, image_size, image_size)

    # Create environments
    env = _make_robomimic_image_env(
        dataset_path,
        shape_meta,
        seed=seed,
        image_size=image_size
    )
    eval_env = _make_robomimic_image_env(
        dataset_path,
        shape_meta,
        seed=seed + 1000,
        image_size=image_size
    )

    # Load dataset
    dataset = _load_robomimic_image_dataset(
        dataset_path,
        shape_meta,
        image_size=image_size
    )

    return env, eval_env, dataset, shape_meta


def _check_dataset_exists_image(task, dataset_type):
    """Check if image dataset exists and return path."""
    # Try image dataset first
    file_name = "image_v141.hdf5"
    download_folder = os.path.join(
        expanduser("~/.robomimic"),
        task,
        dataset_type,
        file_name
    )

    if os.path.exists(download_folder):
        return download_folder

    # Fallback to low_dim dataset (can still extract images from it)
    if dataset_type == "mg":
        file_name = "low_dim_sparse_v141.hdf5"
    else:
        file_name = "low_dim_v141.hdf5"

    download_folder = os.path.join(
        expanduser("~/.robomimic"),
        task,
        dataset_type,
        file_name
    )

    if not os.path.exists(download_folder):
        raise FileNotFoundError(
            f"Dataset not found at {download_folder}. "
            f"Please download the dataset first."
        )

    return download_folder


def _make_robomimic_image_env(dataset_path, shape_meta, seed=0, image_size=None):
    """Create a single robomimic image environment."""
    # Get environment metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Disable object state observation for image mode
    env_meta["env_kwargs"]["use_object_obs"] = False

    # Initialize observation modality mapping
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # Create base environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )

    # Wrap with custom wrapper
    env = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta,
        init_state=None,
        image_size=image_size,
    )

    env.seed(seed)

    return env


def _load_robomimic_image_dataset(dataset_path, shape_meta, image_size=None):
    """Load robomimic image dataset with lazy loading to save memory.

    Args:
        dataset_path: Path to HDF5 dataset
        shape_meta: Shape metadata dictionary
        image_size: Size to resize images to (if None, use original size)

    Returns:
        Dataset object with lazy-loaded observations
    """
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        num_demos = len(demos)
        inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
        demos = [demos[i] for i in inds]

        # Count total timesteps
        num_timesteps = 0
        for ep in demos:
            num_timesteps += int(f[f"data/{ep}/actions"].shape[0])

        print(f"Loading image dataset with {num_timesteps} timesteps from {num_demos} demos")

        # Data holders - load non-image data into memory
        actions = []
        terminals = []
        rewards = []
        masks = []

        # Store episode info for lazy loading
        episode_starts = []
        episode_lengths = []
        current_idx = 0

        # Load non-image data from each demo
        for ep in demos:
            # Load actions
            a = np.array(f[f"data/{ep}/actions"])
            ep_len = len(a)
            actions.append(a.astype(np.float32))

            episode_starts.append(current_idx)
            episode_lengths.append(ep_len)
            current_idx += ep_len

            # Load rewards and dones
            r = np.array(f[f"data/{ep}/rewards"])
            dones = np.array(f[f"data/{ep}/dones"])
            rewards.append(r.astype(np.float32))
            terminals.append(dones.astype(np.float32))
            masks.append(1.0 - dones.astype(np.float32))

        # Concatenate non-image data
        actions_array = np.concatenate(actions, axis=0)
        rewards_array = np.concatenate(rewards, axis=0)
        terminals_array = np.concatenate(terminals, axis=0)
        masks_array = np.concatenate(masks, axis=0)

    # Create dataset with lazy loading for images
    from utils.datasets import LazyImageDataset

    return LazyImageDataset.create(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        actions=actions_array,
        rewards=rewards_array,
        terminals=terminals_array,
        masks=masks_array,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
        image_size=image_size,
    )


def _process_images(images, target_size=None):
    """Process images: normalize to [0, 1] and optionally resize.

    Args:
        images: Array of shape (N, H, W, C) or (N, C, H, W)
        target_size: Target size (H, W) to resize to

    Returns:
        Processed images in (N, C, H, W) format, normalized to [0, 1]
    """
    # Normalize to [0, 1]
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    # Convert to (N, C, H, W) if needed
    if images.ndim == 4:
        if images.shape[-1] in [1, 3]:  # (N, H, W, C)
            images = np.transpose(images, (0, 3, 1, 2))

    # Resize if needed
    if target_size is not None:
        try:
            import cv2
            n, c, h, w = images.shape
            if (h, w) != (target_size, target_size):
                resized = np.zeros((n, c, target_size, target_size), dtype=images.dtype)
                for i in range(n):
                    for j in range(c):
                        resized[i, j] = cv2.resize(
                            images[i, j],
                            (target_size, target_size),
                            interpolation=cv2.INTER_AREA
                        )
                images = resized
        except ImportError:
            print("Warning: cv2 not available, skipping image resize")

    return images


class RobomimicImageWrapper(gym.Env):
    """Wrapper for robomimic image-based environments.

    Handles observation dict formatting and image preprocessing.
    """

    def __init__(
        self,
        env,
        shape_meta,
        init_state=None,
        image_size=None,
        render_obs_key="agentview_image",
    ):
        self.env = env
        self.shape_meta = shape_meta
        self.init_state = init_state
        self.image_size = image_size
        self.render_obs_key = render_obs_key
        self.render_cache = None
        self.has_reset_before = False
        self.seed_state_map = {}
        self._seed = None

        # Setup action space
        action_shape = shape_meta["action"]["shape"]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32
        )

        # Setup observation space (dict)
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if value["type"] == "rgb":
                # Images in [0, 1]
                obs_space = spaces.Box(
                    low=0, high=1, shape=shape, dtype=np.float32
                )
            else:
                # Low-dim observations
                obs_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
                )
            observation_space[key] = obs_space

        self.observation_space = observation_space

    def get_observation(self, raw_obs=None):
        """Extract and process observations from raw environment observation."""
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        obs = {}

        for key in self.observation_space.keys():
            # Map dataset keys to environment keys
            # Dataset has keys like 'agentview_image' but env provides 'agentview'
            env_key = key.replace("_image", "") if key.endswith("_image") else key

            if env_key in raw_obs:
                data = raw_obs[env_key]

                # Process images
                if self.shape_meta["obs"][key]["type"] == "rgb":
                    data = self._process_single_image(data)

                obs[key] = data
            else:
                # Create zero array if key not found
                obs[key] = np.zeros(
                    self.observation_space[key].shape,
                    dtype=self.observation_space[key].dtype
                )

        # Cache render image
        render_key = self.render_obs_key.replace("_image", "")
        if render_key in raw_obs:
            self.render_cache = raw_obs[render_key]

        return obs

    def _process_single_image(self, img):
        """Process a single image observation."""
        # Normalize to [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Convert to (C, H, W) if needed
        if img.ndim == 3 and img.shape[-1] in [1, 3]:  # (H, W, C)
            img = np.transpose(img, (2, 0, 1))

        # Resize if needed
        if self.image_size is not None:
            try:
                import cv2
                c, h, w = img.shape
                if (h, w) != (self.image_size, self.image_size):
                    resized = np.zeros((c, self.image_size, self.image_size), dtype=img.dtype)
                    for i in range(c):
                        resized[i] = cv2.resize(
                            img[i],
                            (self.image_size, self.image_size),
                            interpolation=cv2.INTER_AREA
                        )
                    img = resized
            except ImportError:
                pass

        return img

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            self._seed = seed

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            self.seed(seed)

        if self.init_state is not None:
            if not self.has_reset_before:
                # Full reset at least once for correct rendering
                self.env.reset()
                self.has_reset_before = True

            raw_obs = self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                # Use cached state
                raw_obs = self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # Full reset and cache state
                np.random.seed(seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # Random reset
            raw_obs = self.env.reset()

        obs = self.get_observation(raw_obs)
        return obs, {}

    def step(self, action):
        """Step environment."""
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)

        # Convert to gymnasium API (5 returns)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        """Render environment."""
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")

        # Convert from (C, H, W) to (H, W, C) and scale to [0, 255]
        img = np.moveaxis(self.render_cache, 0, -1)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        return img


if __name__ == "__main__":
    # Test image environment creation
    env, eval_env, dataset, shape_meta = make_env_and_dataset_image(
        "lift-mh-image",
        seed=0,
        image_size=84,
        use_eye_in_hand=True
    )

    print("Shape meta:", shape_meta)
    print("Dataset size:", len(dataset))
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Test reset
    obs, info = env.reset()
    print("Observation keys:", obs.keys())
    for key, value in obs.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
