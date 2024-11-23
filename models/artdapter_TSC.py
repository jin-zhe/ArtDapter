from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class AdaLayerNorm(nn.Module):
	def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
		super().__init__()

		if time_embedding_dim is None:
			time_embedding_dim = embedding_dim

		self.silu = nn.SiLU()
		self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
		nn.init.zeros_(self.linear.weight)
		nn.init.zeros_(self.linear.bias)

		self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

	def forward(
		self, x: torch.Tensor, timestep_embedding: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		emb = self.linear(self.silu(timestep_embedding))
		shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
		x = self.norm(x) * (1 + scale) + shift
		return x


class SquaredReLU(nn.Module):
	def forward(self, x: torch.Tensor):
		return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
	def __init__(
		self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
	):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

		self.mlp = nn.Sequential(
			OrderedDict(
				[
					("c_fc", nn.Linear(d_model, d_model * 4)),
					("sq_relu", SquaredReLU()),
					("c_proj", nn.Linear(d_model * 4, d_model)),
				]
			)
		)

		self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
		self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
		self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

	def attention(self, q: torch.Tensor, kv: torch.Tensor):
		attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
		return attn_output

	def forward(
		self,
		x: torch.Tensor,
		latents: torch.Tensor,
		timestep_embedding: torch.Tensor = None,
	):
		normed_latents = self.ln_1(latents, timestep_embedding)
		latents = latents + self.attention(
			q=normed_latents,
			kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
		)
		latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
		return latents


class PerceiverResampler(nn.Module):
	def __init__(
		self,
		width: int = 768,
		num_blocks: int = 6,
		heads: int = 8,
		num_latents: int = 64,
		output_dim=None,
		input_dim=None,
		time_embedding_dim: Optional[int] = None,
	):
		super().__init__()
		self.output_dim = output_dim
		self.input_dim = input_dim
		self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
		self.time_aware_linear = nn.Linear(
			time_embedding_dim or width, width, bias=True
		)

		if self.input_dim is not None:
			self.proj_in = nn.Linear(input_dim, width)

		self.perceiver_blocks = nn.Sequential(
			*[
				PerceiverAttentionBlock(
					width, heads, time_embedding_dim=time_embedding_dim
				)
				for _ in range(num_blocks)
			]
		)

		if self.output_dim is not None:
			self.proj_out = nn.Sequential(
				nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
			)

	def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
		learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
		latents = learnable_latents + self.time_aware_linear(
			torch.nn.functional.silu(timestep_embedding)
		)
		if self.input_dim is not None:
			x = self.proj_in(x)
		for p_block in self.perceiver_blocks:
			latents = p_block(x, latents, timestep_embedding=timestep_embedding)

		if self.output_dim is not None:
			latents = self.proj_out(latents)

		return latents # size: (b, num_latents, width) e.g. (b, 64, 768)


class ArtDapterTSC(nn.Module):
	NUM_PRINCIPLES = 10

	def __init__(
		self,
		time_channel=320,
		time_embed_dim=768,
		act_fn: str = "silu",
		out_dim: Optional[int] = None,
		connector_width=768,
		connector_blocks=6,
		connector_heads=8,
		connector_latents=64,
		connector_in_dim=2048,
		cast_dtype=None
	):
		super().__init__()

		self.time_channel = time_channel
		self.time_embed_dim = time_embed_dim
		self.act_fn = act_fn
		self.out_dim = out_dim
		self.connector_width = connector_width
		self.connector_blocks = connector_blocks
		self.connector_heads = connector_heads
		self.connector_latents = connector_latents
		self.connector_in_dim = connector_in_dim
		self.cast_dtype = cast_dtype
		self.position = Timesteps(
			self.time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
		)
		self.time_embedding = TimestepEmbedding(
			in_channels =			self.time_channel,
			time_embed_dim =	self.time_embed_dim,
			act_fn =					self.act_fn,
			out_dim =					self.out_dim,
		)

		self.connector = PerceiverResampler(
			width =	self.connector_width,
			num_blocks =	self.connector_blocks,
			heads =	self.connector_heads,
			num_latents = self.connector_latents,
			input_dim = self.connector_in_dim,
			time_embedding_dim = self.time_embed_dim,
		)


	def forward(self, input, timestep):
		device = input.device
		dtype = input.dtype
		batch_size = input.size(0)

		ori_time_feature = self.position(timestep.view(-1)).to(device, dtype=dtype)
		ori_time_feature = ori_time_feature.unsqueeze(dim=1) if ori_time_feature.ndim == 2 else ori_time_feature
		ori_time_feature = ori_time_feature.expand(batch_size, -1, -1)
		time_embedding = self.time_embedding(ori_time_feature)

		out = self.connector(input, timestep_embedding=time_embedding) # size: (b, connector_latents, connector_width)
		if self.cast_dtype:
			out = out.dtype(self.cast_dtype)
		return out
