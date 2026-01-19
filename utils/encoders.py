"""Visual observation encoders for image-based tasks.

Ported from much-ado-about-noising repository.
Supports ResNet-based visual encoders and multi-modal observations.
"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as ttf


def at_least_ndim(x, ndim):
    """Ensure tensor has at least ndim dimensions by adding trailing dimensions."""
    while x.ndim < ndim:
        x = x.unsqueeze(-1)
    return x


class IdentityEncoder(nn.Module):
    """Identity encoder that passes through the input with optional dropout."""

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor | dict, mask: torch.Tensor = None):
        """
        Args:
            condition: (batch, *shape) tensor or dict of tensors
            mask: (batch,) mask tensor or None
        Returns:
            condition with mask applied
        """
        # Handle dict input by concatenating all tensors
        if isinstance(condition, dict):
            keys = sorted(condition.keys())
            tensors = [condition[k] for k in keys]
            flattened = [t.reshape(t.shape[0], -1) for t in tensors]
            condition = torch.cat(flattened, dim=-1)

        # Apply mask
        if mask is None:
            if self.training:
                mask = (torch.rand(condition.shape[0], device=condition.device) > self.dropout).float()
            else:
                mask = 1.0

        mask = at_least_ndim(mask, condition.ndim)
        return condition * mask


class MLPEncoder(nn.Module):
    """MLP encoder for low-dimensional observations."""

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        To: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.To = To
        self.emb_dim = emb_dim

        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims

        # Build MLP
        layers = []
        in_dim = obs_dim * To
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                act,
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, emb_dim * To))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        """
        Args:
            obs: (batch, To, obs_dim) tensor or dict of tensors
            mask: (batch,) mask tensor or None
        Returns:
            (batch, To, emb_dim) encoded observations
        """
        # Handle dict input
        if isinstance(obs, dict):
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
        else:
            obs_list = [obs]

        # Flatten and concatenate
        flattened = [t.reshape(t.shape[0], -1) for t in obs_list]
        obs = torch.cat(flattened, dim=-1)

        # Apply mask
        if mask is None:
            if self.training:
                mask = (torch.rand(obs.shape[0], device=obs.device) > self.dropout).float()
            else:
                mask = 1.0

        mask = at_least_ndim(mask, obs.ndim)
        emb_features = self.mlp(obs) * mask
        return emb_features.reshape(obs.shape[0], self.To, self.emb_dim)


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace submodules matching predicate with func output."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def get_resnet(name, weights=None, **kwargs):
    """Get ResNet model with identity final layer."""
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class CropRandomizer(nn.Module):
    """Randomly crop images during training, center crop during eval."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        super().__init__()
        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward(self, inputs):
        """Apply random crops during training, center crop during eval."""
        if self.training:
            # Random crop
            B, C, H, W = inputs.shape
            # Simple random crop implementation
            h_start = torch.randint(0, H - self.crop_height + 1, (B,), device=inputs.device)
            w_start = torch.randint(0, W - self.crop_width + 1, (B,), device=inputs.device)

            crops = []
            for i in range(B):
                crop = inputs[i:i+1, :, h_start[i]:h_start[i]+self.crop_height,
                             w_start[i]:w_start[i]+self.crop_width]
                crops.append(crop)
            return torch.cat(crops, dim=0)
        else:
            # Center crop
            return ttf.center_crop(inputs, (self.crop_height, self.crop_width))


class MultiImageObsEncoder(nn.Module):
    """Multi-modal observation encoder supporting both RGB and low-dim inputs."""

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str,
        emb_dim: int = 256,
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        use_seq=False,
        keep_horizon_dims=False,
    ):
        super().__init__()
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        # Get RGB model
        if "resnet" in rgb_model_name:
            rgb_model = get_resnet(rgb_model_name)
        else:
            raise ValueError(f"Unsupported rgb_model: {rgb_model_name}")

        # Handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape

            if type == "rgb":
                rgb_keys.append(key)
                # Configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # Configure transforms
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform

            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims

        # MLP to project concatenated features to emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def multi_image_forward(self, obs_dict):
        """Process multi-modal observations."""
        batch_size = None
        seq_len = None
        features = []

        # Process RGB inputs
        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if self.use_seq:
                    if batch_size is None:
                        batch_size = img.shape[0]
                        seq_len = img.shape[1]
                    img = img.reshape(batch_size * seq_len, *img.shape[2:])
                else:
                    if batch_size is None:
                        batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            num_keys = len(self.rgb_keys)
            feature_dim = feature.shape[-1]
            feature = feature.view(
                num_keys,
                batch_size if not self.use_seq else batch_size * seq_len,
                feature_dim,
            )
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(
                batch_size if not self.use_seq else batch_size * seq_len, -1
            )
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if self.use_seq:
                    if batch_size is None:
                        batch_size = img.shape[0]
                        seq_len = img.shape[1]
                    img = img.reshape(batch_size * seq_len, *img.shape[2:])
                else:
                    if batch_size is None:
                        batch_size = img.shape[0]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # Process low-dim inputs
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if self.use_seq:
                if batch_size is None:
                    batch_size = data.shape[0]
                    seq_len = data.shape[1]
                data = data.reshape(batch_size * seq_len, *data.shape[2:])
            else:
                if batch_size is None:
                    batch_size = data.shape[0]
            features.append(data)

        # Concatenate all features
        features = torch.cat(features, dim=-1)
        return features, batch_size, seq_len

    def forward(self, obs_dict, mask=None):
        """
        Args:
            obs_dict: Dict of observations
            mask: Optional mask
        Returns:
            Encoded observations
        """
        features, batch_size, seq_len = self.multi_image_forward(obs_dict)
        result = self.mlp(features)

        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.view(batch_size, seq_len, -1)
            else:
                result = result.view(batch_size, -1)
        return result

    @torch.no_grad()
    def output_shape(self):
        """Compute output shape by running a forward pass."""
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            prefix = (batch_size, 1) if self.use_seq else (batch_size,)
            this_obs = torch.zeros(prefix + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output, _, _ = self.multi_image_forward(example_obs_dict)
        return example_output.shape[1]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class ResnetStack(nn.Module):
    """ResNet stack module for IMPALA encoder."""
    def __init__(self, in_channels, num_features, num_blocks, max_pooling=True):
        super().__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        self.conv_in = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_uniform_(self.conv_in.weight)
        if self.conv_in.bias is not None:
            nn.init.zeros_(self.conv_in.bias)

        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'conv1': nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                'conv2': nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
            })
            for _ in range(num_blocks)
        ])

        for block in self.blocks:
            for name, layer in block.items():
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.output_dim = num_features

    def forward(self, x):
        conv_out = self.conv_in(x)

        if self.max_pooling:
            conv_out = self.max_pool(conv_out)

        # Residual Blocks
        for block in self.blocks:
            block_input = conv_out

            out = torch.nn.functional.relu(conv_out)
            out = block['conv1'](out)
            out = torch.nn.functional.relu(out)
            out = block['conv2'](out)

            conv_out = out + block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder for visual observations."""
    def __init__(self,
                 input_shape,  # (C, H, W)
                 width=1,
                 stack_sizes=(16, 32, 32),
                 num_blocks=2,
                 dropout_rate=None,
                 mlp_hidden_dims=(512,),
                 layer_norm=False):
        super().__init__()

        from utils.base_networks import MLP

        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm

        # 1. Build Stacks
        self.stacks = nn.ModuleList()
        current_channels = input_shape[0]

        for i, size in enumerate(stack_sizes):
            out_channels = size * width
            stack = ResnetStack(
                in_channels=current_channels,
                num_features=out_channels,
                num_blocks=num_blocks,
                max_pooling=True
            )
            self.stacks.append(stack)
            current_channels = out_channels

        # 2. Dropout
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        # 3. Calculate Flatten Dim (Dummy Pass)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_out = self._forward_conv(dummy_input)
            self.flatten_dim = dummy_out.reshape(1, -1).shape[1]

        # 4. Layer Norm
        if layer_norm:
            self.ln = nn.LayerNorm(current_channels)
        else:
            self.ln = None

        # 5. MLP
        self.mlp = MLP(
            input_dim=self.flatten_dim,
            action_dim=mlp_hidden_dims[-1],
            hidden_dim=mlp_hidden_dims[:-1] if len(mlp_hidden_dims) > 1 else (),
            activate_final=True
        )

        self.output_dim = mlp_hidden_dims[-1]

    def _forward_conv(self, x):
        # Normalize
        x = x.float() / 255.0
        for stack in self.stacks:
            x = stack(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x

    def forward(self, x, train=True, cond_var=None):
        conv_out = self._forward_conv(x)
        conv_out = torch.nn.functional.relu(conv_out)

        if self.ln is not None:
            # (N, C, H, W) -> (N, H, W, C) for LayerNorm
            conv_out = conv_out.permute(0, 2, 3, 1)
            conv_out = self.ln(conv_out)
            out = conv_out.reshape(conv_out.size(0), -1)
        else:
            out = conv_out.reshape(conv_out.size(0), -1)

        out = self.mlp(out)
        return out
