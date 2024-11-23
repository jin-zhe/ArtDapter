'''
Initializes model weights using the stable-diffusion pre-trained weights
'''
import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer

from utils import count_parameters
from models.util import create_model
from utils import prepare_target_weights
from models.hack import disable_verbosity, enable_sliced_attention


def get_args():
	parser = argparse.ArgumentParser(description='Prepare weights.')
	parser.add_argument('--init_dir',					'-idir',	default='ckpt/init/',	type=Path)
	parser.add_argument('--checkpoint',				'-ckpt',	default='v1-5-pruned.ckpt',	type=str)
	parser.add_argument('--ella_checkpoint',	'-eckpt',	default='ella-sd1.5-tsc-t5xl.safetensors',	type=str)
	parser.add_argument('--config', 					'-cfg',		default='configs/train_config.yaml',				type=str)
	parser.add_argument('--output', 					'-o',			default='init.ckpt',				type=str)
	parser.add_argument('--precision', 				'-p',			default='16-mixed',										type=str,	choices=['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16'])
	parser.add_argument('--gpus',							'-g',			default='0',													type=str)
	parsed = parser.parse_args()

	parsed.checkpoint = parsed.init_dir / parsed.checkpoint
	parsed.ella_checkpoint = parsed.init_dir / parsed.ella_checkpoint
	parsed.output = parsed.init_dir / parsed.output
	parsed.gpus = [int(g.strip()) for g in parsed.gpus.split(',')]
	return parsed


@torch.no_grad
def main():
	disable_verbosity()
	enable_sliced_attention()
	args = get_args()

	trainer = Trainer(accelerator='gpu', devices=args.gpus, precision=args.precision)
	with trainer.init_module():
		model = create_model(config_path=args.config)
		print(f'ArtDapter params: {round(count_parameters(model.artdapter)/1000000000,2)}B')
		target_weights = prepare_target_weights(model, str(args.checkpoint), device='cuda')
		model.load_state_dict(target_weights, strict=True)
	torch.save(model.state_dict(), str(args.output))
	print('Weights initialized!')


if __name__ == '__main__': main()
