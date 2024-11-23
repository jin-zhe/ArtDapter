import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = max(w, h)
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant')


class Unpermute(torch.nn.Module):
	'''Undo the C x H x W permutation done by PILToTensor because it will later be done by DDPM's get_input'''
	def forward(self, image):
		return image.permute(1,2,0)


class CompArt(Dataset):
	SPLITS = ['train', 'test']
	PoA_PRINCIPLES = ['balance', 'harmony', 'variety', 'unity', 'contrast', 'emphasis', 'proportion', 'movement', 'rhythm', 'pattern']

	def __init__(self,
				dataset_path,
				split,
				image_size,
				dtype,
				drop_caption_prob=None,
				drop_art_style_prob=None,
				keep_all_PoA_prob=None,
				drop_all_PoA_prob=None,
				drop_each_PoA_prob=None):

		self.dataset_path = dataset_path
		self.split = split
		self.image_size = image_size
		self.dtype = getattr(torch, dtype)
		self.drop_caption_prob = drop_caption_prob
		self.drop_art_style_prob = drop_art_style_prob
		self.keep_all_PoA_prob = keep_all_PoA_prob
		self.drop_all_PoA_prob = drop_all_PoA_prob
		self.drop_each_PoA_prob = drop_each_PoA_prob

		if self.split not in CompArt.SPLITS:
			raise ValueError(f'Split "{self.split}" does not exist!')

		self.image_transforms = transforms.Compose([
			SquarePad(),
			transforms.Resize(self.image_size),
			transforms.CenterCrop(self.image_size),
			transforms.PILToTensor(),
			transforms.ToDtype(self.dtype, scale=True),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize to range [-1,1]
			Unpermute()
		])
		self.dataset = load_dataset(self.dataset_path, split=self.split).with_transform(self._dataset_transform)


	@staticmethod
	def collate_fn(entries):
		image = torch.stack([e['image'] for e in entries])
		caption = [e['caption'] for e in entries]
		art_style = [e['art_style'] for e in entries]
		PoA = [e['PoA'] for e in entries]
		identifier = [e['identifier'] for e in entries]
		return dict(image=image, caption=caption, art_style=art_style, PoA=PoA, identifier=identifier)

	
	def _dataset_transform(self, dataset):
		dataset['image'] = [self.image_transforms(img.convert("RGB")) for img in dataset['image']]
		return dataset


	def __getitem__(self, index):
		item = self.dataset[index]
		identifier = item['identifier']
		caption = item['caption']
		art_style = item['art_style']
		PoA_analyses = [item['PoA'][p]['analysis'] for p in self.PoA_PRINCIPLES]

		if self.split == 'train':
			if random.random() < self.drop_caption_prob:
				caption = ''
			if random.random() < self.drop_art_style_prob:
				art_style = ''
			PoA = []
			if random.random() < self.keep_all_PoA_prob:		# if keeping all
				PoA = PoA_analyses
			else:
				if random.random() < self.drop_all_PoA_prob:	# if dropping all
					PoA = ['' for p in PoA_analyses]
				else:
					PoA = [('' if random.random() < self.drop_each_PoA_prob else p) for p in PoA_analyses]
		elif self.split == 'test':
			PoA = PoA_analyses

		return dict(image=item['image'], caption=caption, art_style=art_style, PoA=PoA, identifier=identifier)


	def __len__(self):
		return len(self.dataset)
		