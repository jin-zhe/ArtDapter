import math

import wandb
import torch
from tqdm import tqdm
from einops import rearrange, repeat
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .general import wandb_htmltable, get_subbatch


class CustomLogger(Callback):
	def __init__(self, log_frequency, n_examples, unconditional_guidance_scale, sample, use_ddim, ddim_steps, ddim_intermediates, ddim_eta, plot_denoise_rows, plot_diffusion_rows):
		super().__init__()
		self.log_freq = log_frequency
		self.n_examples = n_examples
		self.unconditional_guidance_scale = unconditional_guidance_scale
		self.sample = sample
		self.use_ddim = use_ddim
		self.ddim_steps = ddim_steps
		self.ddim_intermediates = ddim_intermediates
		self.ddim_eta = ddim_eta
		self.plot_denoise_rows = plot_denoise_rows
		self.plot_diffusion_rows = plot_diffusion_rows


	@torch.no_grad()
	def get_log_dict(self, pl_module, log_batch):
		log_dict = dict(
			image =						log_batch[pl_module.first_stage_key],
			caption =					log_batch[pl_module.cond_stage_key],
			art_style =				log_batch['art_style'],
			PoA =							log_batch['PoA'],
			samples =					None,
			samples_cfg =			None,
			diffusion_row =		None,
			denoise_row =			None
		)

		z, cond = pl_module.get_input(log_batch, pl_module.first_stage_key)
		log_dict['reconstruction'] =	pl_module.decode_first_stage(z)

		c_ca =	cond["c_crossattn"][0]
		if pl_module.artdapter_choice == 'ArtDapterTSC':
			cond_dict = dict(c_crossattn=[c_ca])
		else:
			cond_dict = dict(c_crossattn=[c_ca], art_style_control=[cond["art_style_control"][0]], PoA_controls=[cond["PoA_controls"][0]])

		if self.plot_diffusion_rows:
			diffusion_row = list()
			z_start = z
			for t in range(pl_module.num_timesteps):
				if (t % pl_module.log_every_t == 0) or (t == pl_module.num_timesteps - 1):
					t = repeat(torch.tensor([t]), '1 -> b', b=self.n_examples)
					t = t.to(pl_module.device).long()
					noise = torch.randn_like(z_start)
					z_noisy = pl_module.q_sample(x_start=z_start, t=t, noise=noise)
					diffusion_row.append(pl_module.decode_first_stage(z_noisy))

			diffusion_row = torch.stack(diffusion_row)
			diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
			log_dict["diffusion_row"] = [d.clamp(-1., 1.) for d in diffusion_grid]

		if self.sample:
			z_samples, z_intermediates = pl_module.sample_log(
					cond =				cond_dict,
					batch_size =	self.n_examples,
					ddim =				self.use_ddim,
					ddim_steps =	self.ddim_steps,
					eta =					self.ddim_eta,
					log_every_t =	math.ceil(self.ddim_steps / (self.ddim_intermediates - 2)) # -2 because ddim logs initial noise and the first denoise step
				)
			samples = pl_module.decode_first_stage(z_samples)
			log_dict["sample"] = samples.clamp(-1., 1.)
		if self.sample and self.plot_denoise_rows:
			z_intermediates = z_intermediates['x_inter'] if self.use_ddim else z_intermediates # dict_keys(['x_inter', 'pred_x0'])
			denoise_row = [pl_module.decode_first_stage(z_img.to(pl_module.device)) for z_img in tqdm(z_intermediates, desc='Get denoise row')]
			denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
			denoise_row = rearrange(denoise_row, 'n b c h w -> b n c h w')
			log_dict["denoise_row"] = [d.clamp(-1., 1.) for d in denoise_row]

		if self.unconditional_guidance_scale > 1.0:
			uc_cross = pl_module.get_unconditional_conditioning(self.n_examples)
			if pl_module.artdapter_choice == 'ArtDapterTSC':
				uc_full = dict(c_crossattn=[uc_cross])
			else:
				uc_full = dict(
					c_crossattn=[uc_cross],
					art_style_control=[torch.zeros_like(cond_dict['art_style_control'])],
					PoA_controls=[torch.zeros_like(cond_dict['PoA_controls'])]
				)
			z_samples_cfg, _ = pl_module.sample_log(
					cond =													cond_dict,
					batch_size =										self.n_examples,
					ddim =													self.use_ddim,
					ddim_steps =										self.ddim_steps,
					eta =														self.ddim_eta,
					unconditional_guidance_scale =	self.unconditional_guidance_scale,
					unconditional_conditioning =		uc_full
				)
			samples_cfg = pl_module.decode_first_stage(z_samples_cfg)
			log_dict[f"sample_cfg"] = samples_cfg.clamp(-1., 1.)

		return log_dict


	def log_custom(self, log_dict, step):

		# Log conds
		cond_headers = ['Example', 'Caption', 'Art Style', 'PoA Balance', 'PoA Harmony', 'PoA Variety', 'PoA Unity', 'PoA Contrast', 'PoA Emphasis', 'PoA Proportion', 'PoA Movement', 'PoA Rhythm', 'PoA Pattern']	# PoA sequence follows `dataset.PoA_PRINCIPLES` in config
		cond_rows = []
		for i in range(self.n_examples):
			row =		[i+1]
			row +=	[log_dict['caption'][i]]
			row +=	[log_dict['art_style'][i]]
			row +=	log_dict['PoA'][i]
			cond_rows.append(row)
		wandb.log({'training/example_conds': wandb_htmltable(cond_rows, cond_headers)}, step=step)

		# Log images
		for i in range(self.n_examples):
			image_array = [
				wandb.Image(log_dict['image'][i].permute(2,0,1), caption='Image'),											# shape: (3, 512, 512)
				wandb.Image(log_dict['reconstruction'][i], caption='Reconstruction'),										# shape: (3, 512, 512)
			]
			if self.sample:
				image_array.append(wandb.Image(log_dict['sample'][i], caption='Sample'))								# shape: (3, 512, 512)
			if self.unconditional_guidance_scale > 1.0:
				image_array.append(wandb.Image(log_dict['sample_cfg'][i], caption='Sample cfg'))				# shape: (3, 512, 512)
			if self.plot_diffusion_rows:
				image_array.append(wandb.Image(log_dict['diffusion_row'][i], caption='Diffusion row'))	# shape: (6, 3, 512, 512)
			if self.sample and self.plot_denoise_rows:
				image_array.append(wandb.Image(log_dict['denoise_row'][i], caption='Denoise row'))			# shape: (3, 3, 512, 512)
			wandb.log({f'training/example_{i+1}': image_array}, step=step)


	@rank_zero_only
	def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
		if (self.n_examples > 0) and (pl_module.global_step % self.log_freq == 0):
			is_train = pl_module.training	# cache current context
			if is_train:
				pl_module.eval()

			log_batch = get_subbatch(batch, self.n_examples)
			log_dict = self.get_log_dict(pl_module, log_batch)
			self.log_custom(log_dict, pl_module.global_step)

			if is_train:
				pl_module.train()
