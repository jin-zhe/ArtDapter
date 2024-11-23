import re
import torch
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion


class BaselineModel(LatentDiffusion):

	def __init__(self, *args, adapter_config=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.adapter = None

		if adapter_config is not None:
			self.adapter = instantiate_from_config(adapter_config)


	@torch.no_grad()
	def get_input(self, batch, k, bs=None, *args, **kwargs):
		batch_caption = batch[self.cond_stage_key]
		batch_art_style = batch['art_style']
		batch_PoA = batch['PoA']
		if bs is not None:
			batch_caption = batch_caption[:bs]
			batch_art_style = batch_art_style[:bs]
			batch_PoA = batch_PoA[:bs]

		batch[self.cond_stage_key] = self.format_caption(batch_caption, batch_art_style, batch_PoA)
		z, caption_cond = super().get_input(batch, self.first_stage_key, *args, bs=bs, **kwargs)
		return z, dict(c_crossattn=[caption_cond])


	def format_caption(self, batch_caption, batch_art_style, batch_PoA):
		def format_example(index):
			prompt_seq = [batch_caption[index], batch_art_style[index]+'.'] + batch_PoA[index]
			return re.sub(' +', ' ', ' '.join(prompt_seq).strip())
		out = [format_example(i) for i,_ in enumerate(batch_caption)]
		return out


	def apply_model(self, x_noisy, t, cond, *args, **kwargs):
		cond_caption = torch.cat(cond['c_crossattn'], 1)
		if self.adapter is not None:
			cond_caption = self.adapter(cond_caption, t)
		return self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_caption)


	@torch.no_grad()
	def get_unconditional_conditioning(self, n):
		return self.get_learned_conditioning([""] * n)


	def low_vram_shift(self, mode, device):
		if mode == 'first_stage':
			self.model = self.model.cpu()
			self.first_stage_model = self.first_stage_model.to(device)
			self.cond_stage_model = self.cond_stage_model.cpu()
			if self.adapter:
				self.adapter = self.adapter.cpu()
		elif mode == 'cond_stage':
			self.model = self.model.cpu()
			self.first_stage_model = self.first_stage_model.cpu()
			self.cond_stage_model = self.cond_stage_model.to(device)
			if self.adapter:
				self.adapter = self.adapter.to(device)
		elif mode == 'diffuse_stage':
			self.model = self.model.to(device)
			self.first_stage_model = self.first_stage_model.cpu()
			self.cond_stage_model = self.cond_stage_model.cpu()
			if self.adapter:
				self.adapter = self.adapter.to(device)
