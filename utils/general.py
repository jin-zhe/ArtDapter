import io
import sys
import json
import base64
import warnings
import traceback
from pathlib import Path

import wandb
import torch
import numpy as np
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only

GB_bytes = 1024**3


def freeze(model):
	model.eval()
	for param in model.parameters():
		param.requires_grad = False


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(device):
	if isinstance(device, torch.device):
		return device
	if device == 'cpu':
		return torch.device(type='cpu')
	if type(device) is int:
		return torch.device(type='cuda', index=device)


def prepare_target_weights(model, init_weights_path, device='cpu'):
	'''
	Merge pre-trained weights with random initialized weights
	'''
	pretrained_weights = load_weights(init_weights_path, device=device)
	rand_weights = model.state_dict()
	target_weights = {}
	for state_key in rand_weights.keys():
		if state_key in pretrained_weights.keys():
			target_weights[state_key] = pretrained_weights[state_key].clone()
		else:
			target_weights[state_key] = rand_weights[state_key].clone()
			print(f'New params: {state_key}')
	return target_weights


def load_weights(sd_weights_path, device):
	weights = torch.load(sd_weights_path, map_location=resolve_device(device))
	if 'state_dict' in weights:
		weights = weights['state_dict']
	return weights


def print_gpu_stats(cuda_device):
	print(f"Memory Allocated: {torch.cuda.memory_allocated(cuda_device)/GB_bytes:.2f}GB")
	print(f"Memory Reserved: {torch.cuda.memory_reserved(cuda_device)/GB_bytes:.2f}GB")
	print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(cuda_device)/GB_bytes:.2f}GB")
	print(f"Memory utilized: {torch.cuda.utilization(cuda_device)/GB_bytes:.2f}GB")


def set_warn_with_traceback():
	def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
		log = file if hasattr(file,'write') else sys.stderr
		traceback.print_stack(file=log)
		log.write(warnings.formatwarning(message, category, filename, lineno, line))
	warnings.showwarning = warn_with_traceback


def wandb_htmltable(data, columns, css=''):
	# Helper functions
	def check_shape():
		num_cols = len(columns)
		for i,row in enumerate(data):
			if len(row) != num_cols:
				raise ValueError(f'Row {i} in data should have {num_cols} cols, but found to have {len(row)}.')
	def join(list_str):
		return "".join(list_str)
	def wrap_html_tag(tag, content):
		html_content = content
		inline_style = ''
		if type(content) == dict:
			html_content = content['content']
			inline_style = content.get('style', inline_style)
		if Image.isImageType(html_content):	# if PIL image
			html_content = format_image(html_content, inline_style)
		return f'<{tag} style="{inline_style}">{html_content}</{tag}>'
	def format_image(pil_image, inline_style):
		img_byte_arr = io.BytesIO()
		pil_image.save(img_byte_arr, format='PNG')
		btyestr = base64.b64encode(img_byte_arr.getvalue()).decode()
		return f'<img style="{inline_style}" src="data:image/png;base64,{btyestr}" />'
	def format_html_tr(list_data, is_header=False):
		return wrap_html_tag('tr', join([format_html_td(cd, is_header) for cd in list_data]))
	def format_html_td(cell_data, is_header):
		return wrap_html_tag('th' if is_header else 'td', cell_data)

	check_shape()
	if not css:
		css = wrap_html_tag('style', '''
		table, th, td {
			border: 1px solid black;
			border-collapse: collapse;
		}''')
	column_html = format_html_tr(columns, is_header=True)
	rows_html = join([format_html_tr(row) for row in data])
	table_html = wrap_html_tag('table', join([column_html, rows_html]))
	body_html = wrap_html_tag('body', table_html)
	head_html = wrap_html_tag('head', css)
	iframe_html = wrap_html_tag('html', join([head_html, body_html]))
	return wandb.Html(iframe_html, inject=True)


def get_subbatch(batch, subbatch_size):
	subbatch = dict()
	for key,val in batch.items():
		subbatch[key] = val[:subbatch_size]
	return subbatch


@rank_zero_only
def rank_zero_call(module, function_name, *args):
	return getattr(module, function_name)(*args)


def load_json(json_path: Path):
  '''
  Loads JSON from given path
  '''
  with json_path.open() as f:
    return json.load(f)


def save_json(obj, json_path: Path):
  '''
  Saves given object `obj` as JSON file
  '''
  with json_path.open('w') as f:
    json.dump(obj, f)


def tensor2img(tensor):
	# Expects tensor to be (b, h, w, c)
	return Image.fromarray((tensor.numpy()* 255).astype(np.uint8))
