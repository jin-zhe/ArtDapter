import torch
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion


class ArtDaptedModel(LatentDiffusion):
	PROMPT_TEMPLATE_PREFIXES = ['Prompt', 'Style', 'Balance', 'Harmony', 'Variety', 'Unity', 'Contrast', 'Emphasis', 'Proportion', 'Movement', 'Rhythm', 'Pattern']


	def __init__(self, artdapter_config, *args, art_style_strength=1, PoA_strength=1, **kwargs):
		super().__init__(*args, **kwargs)
		self.artdapter = instantiate_from_config(artdapter_config)
		self.art_style_strength = art_style_strength
		self.PoA_strength = PoA_strength

		self.text_embedding_choice = self.cond_stage_model.__class__.__name__
		self.artdapter_choice = self.artdapter.__class__.__name__


	@torch.no_grad()
	def get_input(self, batch, k, bs=None, *args, **kwargs):
		batch_caption = batch[self.cond_stage_key]
		batch_art_style = batch['art_style']
		batch_PoA = batch['PoA']
		if bs is not None:
			batch_caption = batch_caption[:bs]
			batch_art_style = batch_art_style[:bs]
			batch_PoA = batch_PoA[:bs]
		batch[self.cond_stage_key] = self.apply_prompt_template(batch_caption, batch_art_style, batch_PoA)

		z, caption_cond = super().get_input(batch, self.first_stage_key, *args, bs=bs, **kwargs)

		return z, dict(c_crossattn=[caption_cond])


	def apply_prompt_template(self, batch_caption, batch_art_style, batch_PoA):
		def format_example(index):
			prompt = [batch_caption[index], batch_art_style[index]+'.'] + batch_PoA[index]
			prompt = [f'{self.PROMPT_TEMPLATE_PREFIXES[i]}: {x if x else "None."}' for i,x in enumerate(prompt)]
			return ' '.join(prompt) # e.g. Prompt: Two overlapping monochromatic images of a man in a standing pose with a gun in his hand. Style: Pop Art. Balance: Asymmetric balance is evident in the composition with the two overlapping figures creating a sense of visual weight on the left side, balanced by the lighter area on the right. Harmony: Harmony is achieved through the consistent monochromatic color scheme and the repetition of the same figure, creating a cohesive visual experience. Variety: Variety is introduced by the slight differences in the overlapping figures, adding visual interest without disrupting the overall harmony. Unity: None. Contrast: Contrast is present between the dark figures and the lighter background, helping to highlight the subjects. Emphasis: None. Proportion: None. Movement: None. Rhythm: None. Pattern: None.
		out = [format_example(i) for i,_ in enumerate(batch_caption)]
		return out


	def get_PoA_cond(self, batch_PoA):	# NOT USED. LLAMA decoder code kept for reference
		flattened_PoA = [p for PoA in batch_PoA for p in PoA]	# list of batch_size * 10 * ''
		if self.text_embedding_choice == 'FrozenCLIPEmbedder':
			flattened_PoA_embeddings = self.cond_stage_model(flattened_PoA, layer='pooled').squeeze() # Shape: (batch_size * 10, clip_dim)
			PoA_cond = rearrange(flattened_PoA_embeddings, '(b p) d -> b p d', b=len(batch_PoA), p=self.artdapter.NUM_PRINCIPLES)	# (batch_size, 10, clip_dim)
		elif self.text_embedding_choice == 'FrozenLlamaDecoder':
			flattened_PoA_embeddings = self.cond_stage_model(flattened_PoA) # Shape: (batch_size * 10, tokens, 4096)
			PoA_cond = rearrange(flattened_PoA_embeddings, '(b p) t d -> b p t d', b=len(batch_PoA), p=self.artdapter.NUM_PRINCIPLES)	# (batch_size, 10, tokens, 4096)
		return PoA_cond.to(self.device).to(memory_format=torch.contiguous_format).float()


	def apply_model(self, x_noisy, t, cond, *args, **kwargs):
		cond_caption = torch.cat(cond['c_crossattn'], 1) # shape: (b, num_tokens, clip_dim), e.g. (4, 77, 768)
		cond_caption = self.artdapter(cond_caption, t)
		eps = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_caption)
		return eps


	@torch.no_grad()
	def get_unconditional_conditioning(self, n):
		uc_prompt = 'Prompt: None. Style: . Balance: None. Harmony: None. Variety: None. Unity: None. Contrast: None. Emphasis: None. Proportion: None. Movement: None. Rhythm: None. Pattern: None.'
		return self.get_learned_conditioning([uc_prompt] * n)


	def configure_optimizers(self):
		params = []
		params += list(self.artdapter.parameters())
		if not self.sd_locked:
			params += list(self.model.diffusion_model.output_blocks.parameters())
			params += list(self.model.diffusion_model.out.parameters())
		opt = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
		return opt


	def low_vram_shift(self, mode, device):
		if mode == 'first_stage':
			self.model = self.model.cpu()
			self.first_stage_model = self.first_stage_model.to(device)
			self.cond_stage_model = self.cond_stage_model.cpu()
			if self.artdapter:
				self.artdapter = self.artdapter.cpu()
		elif mode == 'cond_stage':
			self.model = self.model.cpu()
			self.first_stage_model = self.first_stage_model.cpu()
			self.cond_stage_model = self.cond_stage_model.to(device)
			if self.artdapter:
				self.artdapter = self.artdapter.to(device)
		elif mode == 'diffuse_stage':
			self.model = self.model.to(device)
			self.first_stage_model = self.first_stage_model.cpu()
			self.cond_stage_model = self.cond_stage_model.cpu()
			if self.artdapter:
				self.artdapter = self.artdapter.to(device)
