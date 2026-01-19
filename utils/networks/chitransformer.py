"""ChiTransformer architecture for flow matching.

Ported from much-ado-about-noising repository.
Transformer architecture from Diffusion Policy by Chi et al.
"""

import copy

import torch
import torch.nn as nn

from utils.embeddings import SUPPORTED_TIMESTEP_EMBEDDING


class _SimpleTransformerEncoder(nn.Module):
    """Simple Transformer Encoder without nn.TransformerEncoder for compile compatibility."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


class _SimpleTransformerDecoder(nn.Module):
    """Simple Transformer Decoder without nn.TransformerDecoder for compile compatibility."""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return output


def _init_weights(module):
    """Weight initialization for transformer components."""
    ignore_types = (
        nn.Dropout,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.ModuleList,
        nn.Mish,
        nn.Sequential,
    )
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        weight_names = [
            "in_proj_weight",
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
        ]
        for name in weight_names:
            weight = getattr(module, name)
            if weight is not None:
                torch.nn.init.normal_(weight, mean=0.0, std=0.02)

        bias_names = ["in_proj_bias", "bias_k", "bias_v"]
        for name in bias_names:
            bias = getattr(module, name)
            if bias is not None:
                torch.nn.init.zeros_(bias)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
    elif isinstance(module, ignore_types):
        pass


class ChiTransformer(nn.Module):
    """ChiTransformer with flow matching time inputs and scalar prediction head."""

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        Ta: int,
        To: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 8,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        n_cond_layers: int = 0,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        super().__init__()

        # Compute number of tokens for main trunk and condition encoder
        T = Ta
        T_cond = 1 + To  # time + observations

        self.Ta = Ta
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.disable_time_embedding = disable_time_embedding

        # Input embedding stem
        self.input_emb = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, d_model))
        self.drop = nn.Dropout(p_drop_emb)

        # Condition encoder components
        self.cond_obs_emb = nn.Linear(obs_dim, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, d_model))

        # Encoder for conditioning
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = _SimpleTransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.Mish(),
                nn.Linear(4 * d_model, d_model),
            )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = _SimpleTransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )

        # Store mask dimensions
        self.T = T
        self.T_cond = T_cond

        # Decoder head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, act_dim)

        # Mapping for time embeddings (s and t)
        self.timestep_emb_type = timestep_emb_type
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                d_model // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Components for Scalar Head
        self.input_processor = nn.Linear(act_dim, d_model // 4)
        self.final_processor = nn.Linear(d_model, d_model // 4)
        self.scalar_head = nn.Linear(d_model // 4 + d_model // 4 + d_model, 1)

        # Init
        self.apply(_init_weights)
        torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

        # Zero-out scalar head
        nn.init.constant_(self.scalar_head.weight, 0)
        nn.init.constant_(self.scalar_head.bias, 0)

    def _create_masks(self, device):
        """Create attention masks dynamically."""
        # Causal mask for decoder self-attention
        sz = self.T
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
        )

        # Memory mask for decoder cross-attention
        S = self.T_cond
        t_idx, s_idx = torch.meshgrid(
            torch.arange(self.T, device=device),
            torch.arange(S, device=device),
            indexing="ij",
        )
        memory_mask = t_idx >= (s_idx - 1)
        memory_mask = (
            memory_mask.float()
            .masked_fill(memory_mask == 0, float("-inf"))
            .masked_fill(memory_mask == 1, 0.0)
        )

        return mask, memory_mask

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            x: (batch, Ta, act_dim) target action sequence
            s: (batch,) source time parameter
            t: (batch,) target time parameter
            condition: (batch, To, obs_dim) observation sequence condition

        Returns:
            y: (batch, Ta, act_dim) predicted action sequence
            scalar: (batch, 1) predicted scalar value
        """
        b = x.shape[0]
        device = x.device

        if condition is None:
            condition = torch.zeros((b, self.To, self.obs_dim), device=device)

        # Process input for scalar head
        processed_input = self.input_processor(x.mean(dim=1))

        # Prepare time embeddings
        if not torch.is_tensor(s):
            s = torch.tensor([s], dtype=torch.float32, device=device)
        elif len(s.shape) == 0:
            s = s[None].to(device)
        s = s.expand(b)

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        elif len(t.shape) == 0:
            t = t[None].to(device)
        t = t.expand(b)

        # Compute and combine time embeddings
        if not self.disable_time_embedding:
            s_emb = self.map_s(s)
            t_emb = self.map_t(t)
            time_emb = torch.cat([s_emb, t_emb], dim=-1).unsqueeze(1)
        else:
            time_emb = torch.zeros((b, 1, self.d_model), device=device)

        # Process input action sequence
        input_emb = self.input_emb(x)

        # Encoder - process condition sequence
        cond_obs_emb = self.cond_obs_emb(condition)
        cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        cond_input = self.drop(cond_embeddings + position_embeddings)
        memory = self.encoder(cond_input)

        # Decoder - process action sequence
        token_embeddings = input_emb
        t_len = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t_len, :]
        decoder_input = self.drop(token_embeddings + position_embeddings)

        # Create masks
        mask, memory_mask = self._create_masks(device)

        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=mask,
            memory_mask=memory_mask,
        )

        # Predict action sequence
        y = self.ln_f(decoder_output)
        y = self.head(y)

        # Prepare features for scalar head
        processed_final_output = self.final_processor(decoder_output.mean(dim=1))
        scalar_emb = memory.mean(dim=1)

        # Concatenate features for scalar head
        combined_features = torch.cat(
            [processed_input, processed_final_output, scalar_emb], dim=1
        )

        # Predict scalar value
        scalar_output = self.scalar_head(combined_features)

        return y, scalar_output
