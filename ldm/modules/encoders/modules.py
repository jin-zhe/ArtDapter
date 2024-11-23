import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
from ldm.util import count_params


class AbstractEncoder(LightningModule):
	def __init__(self):
		super().__init__()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def encode(self, *args, **kwargs):
		raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

	def encode(self, x):
		return x


class ClassEmbedder(LightningModule):
	def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
		super().__init__()
		self.key = key
		self.embedding = nn.Embedding(n_classes, embed_dim)
		self.n_classes = n_classes
		self.ucg_rate = ucg_rate

	def forward(self, batch, key=None, disable_dropout=False):
		if key is None:
			key = self.key
		# this is for use in crossattn
		c = batch[key][:, None]
		if self.ucg_rate > 0. and not disable_dropout:
			mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
			c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
			c = c.long()
		c = self.embedding(c)
		return c

	def get_unconditional_conditioning(self, bs):
		uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
		uc = torch.ones((bs,), device=self.device) * uc_class
		uc = {self.key: uc}
		return uc


def disabled_train(self, mode=True):
	"""Overwrite model.train with this function to make sure train/eval mode
	does not change anymore."""
	return self


class LlamaDecoder(AbstractEncoder):
	def __init__(self, version="meta-llama/Llama-3.1-8B", max_length=None, freeze=True, out_dtype=None):
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(version)
		self.transformer = AutoModel.from_pretrained(version, device_map="auto", torch_dtype=torch.bfloat16)
		self.max_length = max_length	# default tokenizer max length unless specified otherwise at call time
		self.out_dtype = out_dtype

		self.tokenizer.pad_token = self.tokenizer.eos_token	# specify pad token

		if freeze:
			self.freeze()

	def forward(self, text, max_length):
		max_length = self.max_length if max_length is None else max_length
		tokens = self.tokenizer(
			text,
			max_length=max_length,
			padding="max_length",
			return_tensors="pt").input_ids.to(self.device)
		output = self.transformer(input_ids=tokens, output_hidden_states=True).last_hidden_state
		output = output if self.out_dtype is None else output.dtype(self.out_dtype)
		return output
	
	def encode(self, *args):
		return self(*args)


class T5Embedder(AbstractEncoder):
	def __init__(self, pretrained_path="google/flan-t5-xl", max_length=None, freeze=True):
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
		self.transformer = T5EncoderModel.from_pretrained(pretrained_path, torch_dtype=torch.float16)
		self.max_length = max_length

		if freeze:
			self.freeze()

	def forward(self, text, text_input_ids=None, attention_mask=None, max_length=None):
		if max_length is None:
			max_length = self.max_length

		if text_input_ids is None or attention_mask is None:
			if max_length is not None:
				text_inputs = self.tokenizer(
					text,
					return_tensors="pt",
					add_special_tokens=True,
					max_length=max_length,
					padding="max_length",
					truncation=True,
				)
			else:
				text_inputs = self.tokenizer(
					text, return_tensors="pt", add_special_tokens=True
				)
			text_input_ids = text_inputs.input_ids.to(self.transformer.device)
			attention_mask = text_inputs.attention_mask.to(self.transformer.device)
		outputs = self.transformer(text_input_ids, attention_mask=attention_mask)

		embeddings = outputs.last_hidden_state
		return embeddings.to(dtype=torch.float32)

	def encode(self, *args):
		return self(*args)


class CLIPEmbedder(AbstractEncoder):
	"""Uses the CLIP transformer encoder for text (from huggingface)"""
	LAYERS = [
		"last",
		"pooled",
		"hidden"
	]
	def __init__(self, version="openai/clip-vit-large-patch14", max_length=77,
				 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
		super().__init__()
		assert layer in self.LAYERS
		self.tokenizer = CLIPTokenizer.from_pretrained(version, clean_up_tokenization_spaces=True)
		self.transformer = CLIPTextModel.from_pretrained(version).to(self.device)
		self.max_length = max_length
		if freeze:
			self.freeze()
		self.layer = layer
		self.layer_idx = layer_idx
		if layer == "hidden":
			assert layer_idx is not None
			assert 0 <= abs(layer_idx) <= 12

	def forward(self, text, layer=None):
		layer_choice = self.layer if layer is None else layer
		batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
										return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
		tokens = batch_encoding["input_ids"].to(self.device)
		outputs = self.transformer(input_ids=tokens, output_hidden_states=(layer_choice=="hidden"))
		if layer_choice == "last":
			z = outputs.last_hidden_state
		elif layer_choice == "pooled":
			z = outputs.pooler_output.unsqueeze(1)
		else:
			z = outputs.hidden_states[self.layer_idx]
		return z

	def encode(self, text):
		return self(text)


class OpenCLIPEmbedder(AbstractEncoder):
	"""
	Uses the OpenCLIP transformer encoder for text
	"""
	LAYERS = [
		#"pooled",
		"last",
		"penultimate"
	]
	def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", max_length=77,
				 freeze=True, layer="last"):
		super().__init__()
		assert layer in self.LAYERS
		model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
		del model.visual
		self.model = model

		self.max_length = max_length
		if freeze:
			self.freeze()
		self.layer = layer
		if self.layer == "last":
			self.layer_idx = 0
		elif self.layer == "penultimate":
			self.layer_idx = 1
		else:
			raise NotImplementedError()

	def freeze(self):
		self.model = self.model.eval()
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, text):
		tokens = open_clip.tokenize(text)
		z = self.encode_with_transformer(tokens.to(self.device))
		return z

	def encode_with_transformer(self, text):
		x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
		x = x + self.model.positional_embedding
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.model.ln_final(x)
		return x

	def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
		for i, r in enumerate(self.model.transformer.resblocks):
			if i == len(self.model.transformer.resblocks) - self.layer_idx:
				break
			if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
				x = checkpoint(r, x, attn_mask)
			else:
				x = r(x, attn_mask=attn_mask)
		return x

	def encode(self, text):
		return self(text)


class CLIPT5Encoder(AbstractEncoder):
	def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", clip_max_length=77, t5_max_length=77):
		super().__init__()
		self.clip_encoder = FrozenCLIPEmbedder(clip_version, max_length=clip_max_length)
		self.t5_encoder = FrozenT5Embedder(t5_version, max_length=t5_max_length)
		print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
			  f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

	def freeze(self):
		self.clip_encoder = self.clip_encoder.eval()
		self.t5_encoder = self.t5_encoder.eval()
		for param in self.parameters():
			param.requires_grad = False

	def encode(self, text):
		return self(text)

	def forward(self, text):
		clip_z = self.clip_encoder.encode(text)
		t5_z = self.t5_encoder.encode(text)
		return [clip_z, t5_z]
