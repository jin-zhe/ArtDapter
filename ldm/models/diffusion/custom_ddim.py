import torch

from ldm.modules.diffusionmodules.util import noise_like

from .ddim import DDIMSampler

class CustomDDIMSampler(DDIMSampler):
	@torch.no_grad()
	def sample(self,
							S,
							batch_size,
							shape,
							conditioning = None,
							callback = None,
							normals_sequence = None,
							img_callback = None,
							quantize_x0 = False,
							eta = 0.,
							mask = None,
							x0 = None,
							temperature = 1.,
							noise_dropout = 0.,
							score_corrector = None,
							corrector_kwargs = None,
							verbose = True,
							x_T = None,
							log_every_t = 100,
							unconditional_guidance_scale = 1.,
							unconditional_conditioning = None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
							dynamic_threshold = None,
							ucg_schedule = None,
							global_strength = None,  # only difference
							**kwargs):
		return super().sample(S,
													batch_size,
													shape,
													conditioning = conditioning,
													callback = callback,
													normals_sequence = normals_sequence,
													img_callback = img_callback,
													quantize_x0 = quantize_x0,
													eta = eta,
													mask = mask,
													x0 = x0,
													temperature = temperature,
													noise_dropout = noise_dropout,
													score_corrector = score_corrector,
													corrector_kwargs = corrector_kwargs,
													verbose = verbose,
													x_T = x_T,
													log_every_t = log_every_t,
													unconditional_guidance_scale = unconditional_guidance_scale,
													unconditional_conditioning = unconditional_conditioning,
													dynamic_threshold = dynamic_threshold,
													ucg_schedule = ucg_schedule,
													global_strength = global_strength,
													**kwargs)


	@torch.no_grad()
	def ddim_sampling(self,
										cond,
										shape,
										x_T = None,
										ddim_use_original_steps = False,
										callback = None,
										timesteps = None,
										quantize_denoised = False,
										mask = None,
										x0 = None,
										img_callback = None,
										log_every_t = 100,
										temperature = 1.,
										noise_dropout = 0.,
										score_corrector = None,
										corrector_kwargs = None,
										unconditional_guidance_scale = 1.,
										unconditional_conditioning = None,
										dynamic_threshold = None,
										ucg_schedule = None,
										global_strength = None): # only difference
		return super().ddim_sampling(cond,
																	shape,
																	x_T = x_T,
																	ddim_use_original_steps = ddim_use_original_steps,
																	callback = callback,
																	timesteps = timesteps,
																	quantize_denoised = quantize_denoised,
																	mask = mask,
																	x0 = x0,
																	img_callback = img_callback,
																	log_every_t = log_every_t,
																	temperature = temperature,
																	noise_dropout = noise_dropout,
																	score_corrector = score_corrector,
																	corrector_kwargs = corrector_kwargs,
																	unconditional_guidance_scale = unconditional_guidance_scale,
																	unconditional_conditioning = unconditional_conditioning,
																	dynamic_threshold = dynamic_threshold,
																	ucg_schedule = ucg_schedule,
																	global_strength = global_strength)


	@torch.no_grad()
	def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
										temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
										unconditional_guidance_scale=1., unconditional_conditioning=None,
										dynamic_threshold=None,global_strength=None):
			b, *_, device = *x.shape, x.device

			if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
					model_output = self.model.apply_model(x, t, c)
			else: # only difference in this else-block
					model_t = self.model.apply_model(x, t, c, global_strength)
					model_uncond = self.model.apply_model(x, t, unconditional_conditioning, global_strength)
					model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

			if self.model.parameterization == "v":
					e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
			else:
					e_t = model_output

			if score_corrector is not None:
					assert self.model.parameterization == "eps", 'not implemented'
					e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

			alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
			alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
			sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
			sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
			# select parameters corresponding to the currently considered timestep
			a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
			a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
			sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
			sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

			# current prediction for x_0
			if self.model.parameterization != "v":
					pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
			else:
					pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

			if quantize_denoised:
					pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

			if dynamic_threshold is not None:
					raise NotImplementedError()

			# direction pointing to x_t
			dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
			noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
			if noise_dropout > 0.:
					noise = torch.nn.functional.dropout(noise, p=noise_dropout)
			x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
			return x_prev, pred_x0
