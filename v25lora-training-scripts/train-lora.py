import logging
import random
from pathlib import Path
from typing import Any, Iterator
import math
from collections import defaultdict
from tqdm.contrib.logging import logging_redirect_tqdm

from tqdm import tqdm
from hashlib import sha512

import hydra
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import StableDiffusionXLPipeline
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import itertools
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms.functional as TVF
from torch import nn
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.optim import Optimizer
from transformers import get_scheduler
from datetime import datetime
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
import omegaconf
from hydra.core.config_store import ConfigStore
from diffusers.utils import convert_state_dict_to_diffusers
import tempfile
import safetensors.torch



# SDXL aspect ratio buckets
_AR_BUCKETS = list(range(512, 2049, 64))
_AR_BUCKETS = itertools.product(_AR_BUCKETS, _AR_BUCKETS)
AR_BUCKETS: set[tuple[int, int]] = set([v for v in _AR_BUCKETS if v[0] * v[1] <= 1024*1024 and v[0] * v[1] >= 946176 and v[0]/v[1] >= 0.333 and v[0]/v[1] <= 3.0])


@dataclass
class TrainerConfig:
	seed: int = 69
	batch_size: int = 1
	device_batch_size: int = 1
	total_samples: int = 3000
	base_model: str = omegaconf.MISSING
	base_revision: str | None = None
	base_variant: str | None = None
	train_text_encoder: bool = False
	lora_rank: int = 32
	lora_alpha: int = 32
	gradient_checkpointing: bool = True
	lr: float = 1e-4
	adam_beta1: float = 0.9
	adam_beta2: float = 0.999
	adam_eps: float = 1e-08
	weight_decay: float = 0.0
	fused_optimizer: bool = False
	te_lr_rate: float = 0.5
	cache_dir: Path = Path("latent_cache")
	dataset_dir: Path = Path("dataset")
	checkpoint_dir: Path = Path("checkpoints")
	dataset_num_workers: int = 4
	warmup_samples: int = 100
	lr_scheduler_type: str = "constant_with_warmup" #"cosine"
	min_lr_ratio: float = 0.0
	clip_grad_norm: float | None = 1.0
	noise_schedule: str = "uniform"
	noise_shift: float = 3.0

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)
cs.store(
	group="hydra",       # overrides Hydra’s own defaults
	name="no_io",        # arbitrary name
	node=OmegaConf.create({
		"run": {"dir": "."},        # stay in the original CWD
		"output_subdir": None,      # don’t copy configs to .hydra/
		"job": {"chdir": False},    # don’t chdir into a run dir
		# turn off every file‑logging handler
		"job_logging": "disabled",  # built‑in config: no log file for your job
		"hydra_logging": "disabled" # disable Hydra’s internal log file
	}),
)


class Trainer:
	dataset: "ImageCaptionDataset"
	config: TrainerConfig
	model: "StableDiffusion"

	def __init__(self, config: TrainerConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
		logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		self.logger.setLevel(logging.INFO)

		# TODO: Dump GPU and library versions

		# Performance enhancing drugs
		torch.set_float32_matmul_precision("high")
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

		# Set the random seed for reproducibility
		self.logger.info(f"Setting seed to {config.seed}")
		random.seed(config.seed)
		np.random.seed(config.seed)
		torch.manual_seed(config.seed)
		torch.cuda.manual_seed(config.seed)

		# Calculate device batch size and such
		self.device_batch_size = min(config.batch_size, config.device_batch_size)
		self.gradient_accumulation_steps = config.batch_size // self.device_batch_size
		self.total_steps = config.total_samples // config.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps
		assert config.batch_size == self.device_batch_size * self.gradient_accumulation_steps, f"Batch size {config.batch_size} must be divisible by device batch size {self.device_batch_size} for gradient accumulation steps {self.gradient_accumulation_steps}"

		# Build model
		self.model = StableDiffusion(
			base_model=config.base_model,
			base_revision=config.base_revision,
			base_variant=config.base_variant,
			train_text_encoder=config.train_text_encoder,
			noise_schedule=config.noise_schedule,
			noise_shift=config.noise_shift,
		)
		
		# Apply LORA
		self.model.requires_grad_(False)
		unet_lora_config = LoraConfig(
			r=self.config.lora_rank,
			lora_alpha=self.config.lora_alpha,
			lora_dropout=0.0,
			init_lora_weights=True, #"gaussian",   #config.lora_initialisation_style
			target_modules=["to_k", "to_q", "to_v", "to_out.0", "fc1", "fc2", "k_proj", "v_proj", "q_proj", "out_proj"],   # TE: fc1, fc2, k_proj, v_proj, q_proj, out_proj; additional unet: proj_in, proj_out
		)
		self.model.unet.add_adapter(unet_lora_config)
		if config.train_text_encoder:
			self.model.text_encoder.add_adapter(unet_lora_config)
		total_params = sum(p.numel() for p in self.model.parameters())
		total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.logger.info(f"UNet total parameters: {total_params:,}, trainable parameters: {total_trainable_params:,}")

		if config.gradient_checkpointing:
			self.model.unet.enable_gradient_checkpointing()
		
		# Move to GPU
		self.model.unet.to('cuda')
		self.model.text_encoder.to('cuda')
		self.model.text_encoder_2.to('cuda')

		# Build optimizer
		optimizer_cls = torch.optim.AdamW
		kwargs = {
			'lr': config.lr,
			'betas': (config.adam_beta1, config.adam_beta2),
			'eps': config.adam_eps,
			'weight_decay': config.weight_decay,
			'fused': config.fused_optimizer,
		}

		base_lr = config.lr
		param_groups: list[dict] = [
			{"params": self.model.unet.parameters()},  # Uses base learning rate
		]

		if config.train_text_encoder:
			param_groups.append(
				{"params": self.model.text_encoder.parameters(), "lr": base_lr * config.te_lr_rate}
			)
		
		self.optimizer = optimizer_cls(param_groups, **kwargs)
		self.optimized_params = list(itertools.chain.from_iterable(g['params'] for g in param_groups))

		# Build dataset
		self.train_dataset = ImageCaptionDataset(
			tokenizer=self.model.tokenizer,
			tokenizer_2=self.model.tokenizer_2,
			vae=self.model.vae,
			cache_dir=config.cache_dir,
			dataset_dir=Path(config.dataset_dir),
		)
		self.train_dataset.prepare(self.logger)

		# Build bucket sampler
		buckets = self.train_dataset.gen_buckets()
		self.train_sampler = AspectBucketSampler(dataset=self.train_dataset, buckets=buckets, batch_size=self.device_batch_size, shuffle=True)

		# Build dataloader
		self.train_dataloader = DataLoader(
			self.train_dataset,
			batch_sampler=self.train_sampler,
			num_workers=config.dataset_num_workers,
			collate_fn=self.train_dataset.collate_fn,
			pin_memory=True,
		)
	
		# Build LR schedule
		num_warmup_steps = int(math.ceil(self.config.warmup_samples / self.config.batch_size))
		if self.config.lr_scheduler_type == "cosine":
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
				min_lr_ratio=self.config.min_lr_ratio,
			)
		else:
			self.lr_scheduler = get_scheduler(self.config.lr_scheduler_type, self.optimizer, num_warmup_steps, self.total_steps)
	
	def fit(self):
		device_step = 0
		self.global_step = 0
		self.global_samples_seen = 0

		self.logger.info("Starting training...")
		loss_sum = torch.tensor(0.0, device='cuda', requires_grad=False, dtype=torch.float32)
		dataloader_iter = iter(self.train_dataloader)
		pbar = tqdm(total=self.total_device_batches * self.device_batch_size, initial=0, dynamic_ncols=True, smoothing=0.01)
		with logging_redirect_tqdm():
			for device_step in range(0, self.total_device_batches):
				self.global_step = device_step // self.gradient_accumulation_steps
				self.global_samples_seen = (device_step + 1) * self.device_batch_size

				self.model.eval()
				self.model.unet.train()
				if self.model.train_text_encoder:
					self.model.text_encoder.train()
				
				# Get the next batch
				try:
					batch = next(dataloader_iter)
				except StopIteration:
					self.logger.info("Dataloader exhausted, starting new epoch")
					self.train_sampler.set_epoch(self.train_sampler.epoch + 1)
					dataloader_iter = iter(self.train_dataloader)
					batch = next(dataloader_iter)
				
				# Move batch to device
				batch = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()}

				is_last_device_step = (device_step + 1) % self.gradient_accumulation_steps == 0
				is_last_step = (self.global_step + 1) == self.total_steps

				with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
					# Forward pass
					pred, loss = self.model(batch)
					loss = loss / self.gradient_accumulation_steps
				loss_sum.add_(loss.detach())

				# Backward pass
				loss.backward()

				if is_last_device_step:
					pbar.set_description(f"Loss: {loss_sum.item():.4f}")

					# Clip gradients
					if self.config.clip_grad_norm is not None:
						torch.nn.utils.clip_grad.clip_grad_norm_(self.optimized_params, self.config.clip_grad_norm)
					
					# Take a step
					self.optimizer.step()
					self.lr_scheduler.step()
					self.optimizer.zero_grad(set_to_none=True)

					self.logger.info(f"Samples {self.global_samples_seen} - Loss: {loss_sum.item():.4f} - LR: {self.optimizer.param_groups[0]['lr']:.6f}")
					loss_sum.zero_()
				
				if is_last_step:
					self.save_checkpoint()
			
				pbar.update(self.device_batch_size)
			
			pbar.close()
		
		self.logger.info("Training complete.")
	
	def save_checkpoint(self):
		checkpoint_name = f"checkpoint-{datetime.now().strftime('%Y%m%d_%H%M%S')}.safetensors"

		path = Path(self.config.checkpoint_dir) / checkpoint_name
		self.logger.info(f"Saving checkpoint to {path}")
		tmppath = path.with_suffix('.tmp')
		tmppath.parent.mkdir(parents=True, exist_ok=True)

		# Very convoluted way to save LoRA weights
		# I should probably just write my own save function, but
		# to use the diffusers built in functions this is what it has to be
		with tempfile.TemporaryDirectory() as tmpdir:
			unet_lora_state_dict = convert_state_dict_to_diffusers(
				get_peft_model_state_dict(self.model.unet)
			)
			if self.model.train_text_encoder:
				text_encoder_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.model.text_encoder))
			else:
				text_encoder_lora_state_dict = None

			StableDiffusionXLPipeline.save_lora_weights(
				save_directory=tmpdir,
				unet_lora_layers=unet_lora_state_dict,
				text_encoder_lora_layers=text_encoder_lora_state_dict,
				safe_serialization=True,
			)

			state_dict = safetensors.torch.load_file(Path(tmpdir) / "pytorch_lora_weights.safetensors", device='cpu')
			converted_state_dict, weight_err = convert_to_sd_checkpoint(state_dict)
			self.logger.info(f"Conversion to float16 resulted in a maximum error of {weight_err:.6f}")
			safetensors.torch.save_file(converted_state_dict, tmppath)
			tmppath.rename(path)


################################################################################
#### Model
################################################################################
class StableDiffusion(nn.Module):
	def __init__(
		self,
		base_model: str,
		base_revision: str | None,
		base_variant: str | None,
		train_text_encoder: bool,
		noise_schedule: str,
		noise_shift: float,
		latent_scaling_factor: float = 0.13025,
	):
		super().__init__()

		if base_model.endswith(".safetensors"):
			tmp_model = StableDiffusionXLPipeline.from_single_file(base_model, torch_dtype=torch.float16)
			tokenizer = tmp_model.tokenizer
			tokenizer_2 = tmp_model.tokenizer_2

			text_encoder = tmp_model.text_encoder.to(torch.bfloat16)
			text_encoder_2 = tmp_model.text_encoder_2.to(torch.bfloat16)
			unet = tmp_model.unet.to(torch.bfloat16)
			vae = tmp_model.vae
			tmp_model = None
		else:
			tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=base_revision, use_fast=True)
			tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", revision=base_revision, use_fast=True)

			text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", revision=base_revision, torch_dtype=torch.bfloat16, variant=base_variant, use_safetensors=True)
			text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", revision=base_revision, torch_dtype=torch.bfloat16, variant=base_variant, use_safetensors=True)

			unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", revision=base_revision, torch_dtype=torch.bfloat16, variant=base_variant, use_safetensors=True)
			assert isinstance(unet, UNet2DConditionModel)

			vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", revision=base_revision, torch_dtype=torch.float32, variant=base_variant, use_safetensors=True)

		self.tokenizer = tokenizer
		self.tokenizer_2 = tokenizer_2
		self.text_encoder = text_encoder
		self.text_encoder_2 = text_encoder_2
		self.vae = vae
		self.text_encoder.requires_grad_(train_text_encoder)
		self.text_encoder_2.requires_grad_(False)
		self.train_text_encoder = train_text_encoder
		self.unet = unet
		self.latent_scaling_factor = latent_scaling_factor
		self.noise_schedule = noise_schedule
		self.noise_shift = noise_shift

	def forward(self, batch):
		print(f"Got indices: {batch['idx']}")

		latents: torch.Tensor = batch['latents']
		latents.requires_grad_(False)   # Is this needed?  It was in my original code
		bsz = latents.shape[0]

		# Scale the latents
		latents = latents * self.latent_scaling_factor

		# Encode the prompts
		# batch['prompt'] is expected to be (BxNx77)
		# Squash the first two dimensions to process everything in parallel
		prompt = batch['prompt']
		prompt_2 = batch['prompt_2']
		n = prompt.shape[1]
		assert prompt.shape == (bsz, n, 77), f'Expected prompt shape (bsz, n, 77), got {prompt.shape}'

		with torch.set_grad_enabled(self.train_text_encoder):
			prompt_embed1 = self.text_encoder(prompt.view(-1, 77), output_hidden_states=True).hidden_states[-2]
		
		with torch.no_grad():
			prompt_embed2 = self.text_encoder_2(prompt_2.view(-1, 77), output_hidden_states=True)
			pooled_prompt_embeds = prompt_embed2[0]
			prompt_embed2 = prompt_embed2.hidden_states[-2]
		
		# Unsquash to BxNx77
		prompt_embed1 = prompt_embed1.view(bsz, n*77, prompt_embed1.shape[-1])
		prompt_embed2 = prompt_embed2.view(bsz, n*77, prompt_embed2.shape[-1])
		pooled_prompt_embeds = pooled_prompt_embeds.view(bsz, -1, pooled_prompt_embeds.shape[-1])
		assert prompt_embed1.shape == (bsz, n*77, 768) and prompt_embed2.shape == (bsz, n*77, 1280)
		assert pooled_prompt_embeds.shape == (bsz, n, 1280)

		# Concat the two embeddings along the last dimension (BxN*77x(768+1280))
		text_embeds = torch.concat([prompt_embed1, prompt_embed2], dim=-1)

		# Only the first pooled_prompt_embeds
		# That seems to be how ComfyUI inference does it, but I wonder if there is something better? Average?
		text_pooled_embeds = pooled_prompt_embeds[:, 0, :]
		assert text_pooled_embeds.shape == (bsz, 1280)

		# Prepare added conditioning
		assert batch['original_size'].shape == (bsz, 2)
		add_time_ids = torch.cat((batch['original_size'], batch['crop'], batch['target_size']), dim=1)
		assert add_time_ids.shape == (bsz, 6)

		# Sample the diffusion timesteps
		if self.noise_schedule == "logitnorm":
			sigmas = torch.randn(bsz, device=latents.device, dtype=latents.dtype)
			sigmas = sigmas + math.log(self.noise_shift)
			sigmas = torch.sigmoid(sigmas)
		elif self.noise_schedule == "uniform":
			sigmas = torch.rand(bsz, device=latents.device, dtype=latents.dtype)
		else:
			raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

		# Build noisy samples
		noise = torch.randn(latents.size(), device=latents.device, dtype=latents.dtype)
		sigmas_expanded = sigmas.view(-1, 1, 1, 1)
		noised_latents = (1.0 - sigmas_expanded) * latents + sigmas_expanded * noise
		noised_latents.requires_grad_(False)

		# Generate the targets (flowmatch)
		targets = noise - latents

		# Forward through the model
		pred = self.unet(
			noised_latents,
			sigmas * 1000,
			encoder_hidden_states=text_embeds,
			added_cond_kwargs={
				"text_embeds": text_pooled_embeds,
				"time_ids": add_time_ids,
			}
		).sample

		loss = F.mse_loss(pred, targets, reduction='none')

		# Mask out loss for padding examples
		loss_mask = torch.ones_like(loss, dtype=torch.bool)
		loss_mask[batch['idx'] == -1] = False
		n = loss_mask.sum().item()
		loss_mask = loss_mask.to(loss.device)
		loss = loss * loss_mask.float()
		loss = loss.sum() / n

		return pred, loss


################################################################################
#### Dataset
################################################################################
class ImageCaptionDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		tokenizer: CLIPTokenizer,
		tokenizer_2: CLIPTokenizer,
		vae: AutoencoderKL,
		cache_dir: Path,
		dataset_dir: Path,
	) -> None:
		self.tokenizer = tokenizer
		self.tokenizer_2 = tokenizer_2
		self.vae = vae
		self.cache_dir = cache_dir
		self.dataset_dir = dataset_dir
		self.examples = None
	
	def prepare(self, logger: logging.Logger) -> None:
		"""
		Process the input dataset and cache latents if needed
		"""
		VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

		# Find all examples in the dataset directory
		logger.info(f"Finding images in {self.dataset_dir}...")
		image_paths = [p for p in self.dataset_dir.glob('**/*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
		logger.info(f"Found {len(image_paths)} images in {self.dataset_dir}")
		examples = []

		# Read captions and calculate hashes
		for image_path in tqdm(image_paths, desc="Processing images", dynamic_ncols=True):
			caption_path = image_path.with_suffix('.txt')
			if not caption_path.exists():
				logger.warning(f"Caption file {caption_path} does not exist for image {image_path}, skipping.")
				continue

			caption = caption_path.read_text().strip()
			filehash = sha512(image_path.read_bytes()).digest()[:32]
			filehash_hex = filehash.hex()
			cache_path = self.cache_dir / f"{filehash_hex}.pt"

			example = {
				'filehash': filehash,
				'cache_path': cache_path,
				'caption': caption,
				'file_path': image_path,
			}
			examples.append(example)
		
		# Find any examples that haven't been cached yet and process them
		needs_caching = [ex for ex in examples if not ex['cache_path'].exists()]
		self._run_caching(needs_caching, logger)

		self.examples = examples
		self.vae = None  # Unload VAE to save memory
	
	def _run_caching(self, needs_caching: list[dict[str, Any]], logger: logging.Logger) -> None:
		if len(needs_caching) == 0:
			return
		
		logger.info(f"Found {len(needs_caching)} examples that need caching. Processing...")
		self.vae.eval()
		logger.info(f"VAE scale: {self.vae.config.scaling_factor}")
		self.vae.to('cuda', dtype=torch.float32)

		for example in tqdm(needs_caching, desc="Caching latents", dynamic_ncols=True):
			image = Image.open(example['file_path']).convert('RGB')
			original_width, original_height = image.size
			ar = image.width / image.height

			# Find the AR bucket that is closest to the image's aspect ratio
			# N.B. This is just _one_ way of selecting an "optimal" AR bucket
			# This tends to minimize scaling, but not necessarily cropping
			ar_bucket = min(AR_BUCKETS, key=lambda v: abs(v[0]/v[1] - ar))

			# Scale the image
			scale = max(ar_bucket[0] / image.width, ar_bucket[1] / image.height)
			image = image.resize((int(image.width * scale + 0.5), int(image.height * scale + 0.5)), Image.LANCZOS)
			assert image.width == ar_bucket[0] or image.height == ar_bucket[1]
			assert image.width >= ar_bucket[0] and image.height >= ar_bucket[1]

			# Center crop
			crop_x = (image.width - ar_bucket[0]) // 2
			crop_y = (image.height - ar_bucket[1]) // 2
			cropped = Image.new("RGB", (ar_bucket[0], ar_bucket[1]))
			cropped.paste(image, (-crop_x, -crop_y))

			# Convert to tensor and normalize
			image_tensor = TVF.pil_to_tensor(cropped).to(device='cuda')   # 0-255
			image_tensor = image_tensor / 255.0   # 0-1
			image_tensor = image_tensor - 0.5     # -0.5 to 0.5
			image_tensor = image_tensor * 2.0     # -1.0 to 1.0

			# Encode using the VAE
			latents = self.vae.encode(image_tensor.unsqueeze(0)).latent_dist.sample()
			latents = latents.to(device='cpu', dtype=torch.float32).squeeze(0)  # Remove batch dimension
			assert torch.isfinite(latents).all()

			# Save the latents
			example['cache_path'].parent.mkdir(parents=True, exist_ok=True)
			tmppath = example['cache_path'].with_suffix('.tmp')
			torch.save({
				"latents": latents,
				"original_width": original_width,
				"original_height": original_height,
				"crop_x": crop_x,
				"crop_y": crop_y,
			}, tmppath)
			tmppath.rename(example['cache_path'])
	
	def gen_buckets(self) -> list[list[int]]:
		buckets = defaultdict(list)

		for i, example in tqdm(enumerate(self.examples), desc="Bucketing examples", dynamic_ncols=True):
			cache_data = torch.load(example['cache_path'], map_location='cpu')
			latent_size = cache_data['latents'].shape
			k = (latent_size[1], latent_size[2])
			buckets[k].append(i)
		
		return list(buckets.values())

	def __getitem__(self, index: int):
		# Handle padding sample
		if index == -1:
			return {
				'latents': None,
				'prompt': [],
				'prompt_2': [],
				'original_size': torch.tensor([0, 0], dtype=torch.long),
				'crop': torch.tensor([0, 0], dtype=torch.long),
				'target_size': torch.tensor([0, 0], dtype=torch.long),
				'idx': torch.tensor(-1, dtype=torch.long),
			}

		example = self.examples[index]
		caption = example['caption']

		# Read cached latent
		cache_data = torch.load(example['cache_path'], map_location='cpu')
		latents = cache_data['latents']
		original_width = cache_data['original_width']
		original_height = cache_data['original_height']
		crop_x = cache_data['crop_x']
		crop_y = cache_data['crop_y']

		# Tokenize the prompt
		tokens = self.tokenizer.encode(caption, padding=False, truncation=False, add_special_tokens=False, verbose=False)
		tokens_2 = self.tokenizer_2.encode(caption, padding=False, truncation=False, add_special_tokens=False, verbose=False)

		return {
			'latents': latents,
			'prompt': tokens,
			'prompt_2': tokens_2,
			'original_size': torch.tensor([original_height, original_width], dtype=torch.long),
			'crop': torch.tensor([crop_y, crop_x], dtype=torch.long),
			'target_size': torch.tensor([latents.shape[1] * 8, latents.shape[2] * 8], dtype=torch.long),
			'idx': torch.tensor(index, dtype=torch.long),
		}
	
	def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
		original_sizes = torch.stack([item['original_size'] for item in batch])
		crops = torch.stack([item['crop'] for item in batch])
		target_sizes = torch.stack([item['target_size'] for item in batch])
		idxes = torch.stack([item['idx'] for item in batch])

		# Handle padded latents
		latent_size = next(item['latents'].shape for item in batch if item['latents'] is not None)
		latents = [item['latents'] if item['latents'] is not None else torch.zeros(latent_size, dtype=torch.float32) for item in batch]
		latents = torch.stack(latents)

		# Find the longest prompt length
		# Note: This ignores tokenizer_2, which may have a different length (this is what ComfyUI does)
		n_tokens = max(len(item['prompt']) for item in batch)
		n_chunks = (n_tokens + 74) // 75
		n_chunks = min(max(n_chunks, 1), 3)   # Clamp to 1-3 chunks

		# Chunk up the prompts
		chunks = [chunk_tokens(item['prompt'], n_chunks, self.tokenizer) for item in batch]
		chunks_2 = [chunk_tokens(item['prompt_2'], n_chunks, self.tokenizer_2) for item in batch]

		# Stack the chunks
		chunks = torch.stack(chunks)
		chunks_2 = torch.stack(chunks_2)
		assert chunks.shape == (len(batch), n_chunks, 77) and chunks_2.shape == chunks.shape

		return {
			'latents': latents,
			'prompt': chunks,
			'prompt_2': chunks_2,
			'original_size': original_sizes,
			'crop': crops,
			'target_size': target_sizes,
			'idx': idxes,
		}


def chunk_tokens(tokens: list[int], n_chunks: int, tokenizer: CLIPTokenizer) -> torch.Tensor:
	assert isinstance(tokenizer.bos_token_id, int) and isinstance(tokenizer.eos_token_id, int), f"Tokenizer must have bos_token_id and eos_token_id set and be integers, got {type(tokenizer.bos_token_id)} and {type(tokenizer.eos_token_id)}"
	chunk_tokens = []

	for i in range(0, n_chunks):
		chunk = tokens[i * 75:(i + 1) * 75]
		chunk_tokens.append(tokenizer.bos_token_id)
		chunk_tokens.extend(chunk)
		chunk_tokens.append(tokenizer.eos_token_id)
		chunk_tokens.extend([tokenizer.pad_token_id] * (75 - len(chunk)))
	
	return torch.tensor(chunk_tokens, dtype=torch.long).view(n_chunks, 77)


class AspectBucketSampler(Sampler[list[int]]):
	"""
	Samples batches from a dataset that has been split into aspect ratio buckets.
	Each batch will contain batch_size images from a single bucket
	When the dataset uses randomization, the epoch is meant to be used to deterministically generate the randomization.
	Padding with -1 is used when necessary

	Args:
		dataset: The dataset to sample from.
		buckets: The list of buckets.
		batch_size: The number of images per batch.
		shuffle: Whether to shuffle the images within each bucket.
		seed: The random seed to use for shuffling.
	"""
	def __init__(
		self,
		dataset: Dataset,
		buckets: list[list[int]],
		batch_size: int,
		shuffle: bool = True,
		seed: int = 0,
		ragged_batches: bool = False,
	) -> None:
		self.dataset = dataset
		self.buckets = buckets
		self.epoch = 0
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.seed = seed

		total_batches = sum(int(math.ceil(len(bucket) / batch_size)) for bucket in buckets)
		self.num_samples = total_batches

	def __iter__(self) -> Iterator[list[int]]:
		rng = np.random.default_rng(hash((self.seed, self.epoch)) & 0xFFFFFFFF) if self.shuffle else None

		if rng is not None:
			epoch_buckets = [np.array(bucket, dtype=np.int64) for bucket in self.buckets]

			# Shuffle each bucket
			for bucket in epoch_buckets:
				rng.shuffle(bucket)
		else:
			epoch_buckets = self.buckets
		
		# Split all the buckets into batches
		batches = []

		for bucket in epoch_buckets:
			# Pad to a multiple of batch_size if needed
			if bucket.size % self.batch_size != 0:
				bucket = np.pad(bucket, (0, self.batch_size - (bucket.size % self.batch_size)), mode='constant', constant_values=-1)
			batches.append(bucket.reshape(-1, self.batch_size))
		
		batches = np.concatenate(batches, axis=0)
		
		# Shuffle the batches
		if rng is not None:
			rng.shuffle(batches)
		
		# Flatten and convert to a list of indices
		indices = batches.tolist()

		return iter(indices)
	
	def __len__(self) -> int:
		return self.num_samples
	
	def set_epoch(self, epoch: int) -> None:
		self.epoch = epoch


################################################################################
#### Miscellaneous
################################################################################
def _get_cosine_schedule_with_warmup_lr_lambda(
	current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
	if current_step < num_warmup_steps:
		return float(current_step) / float(max(1, num_warmup_steps))
	
	progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
	r = 1.0 - min_lr_ratio
	return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * r + min_lr_ratio


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.0
):
	lr_lambda = partial(
		_get_cosine_schedule_with_warmup_lr_lambda,
		num_warmup_steps=num_warmup_steps,
		num_training_steps=num_training_steps,
		num_cycles=num_cycles,
		min_lr_ratio=min_lr_ratio,
	)
	return LambdaLR(optimizer, lr_lambda, last_epoch)


################################################################################
### Checkpoint Conversion
################################################################################
unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(3):
	# loop over downblocks/upblocks

	for j in range(2):
		# loop over resnets/attentions for downblocks
		hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
		sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
		unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

		if i > 0:
			hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
			sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
			unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

	for j in range(4):
		# loop over resnets/attentions for upblocks
		hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
		sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
		unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

		if i < 2:
			# no attention layers in up_blocks.0
			hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
			sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
			unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

	if i < 3:
		# no downsample in down_blocks.3
		hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
		sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
		unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

		# no upsample in up_blocks.3
		hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
		sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
		unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
for j in range(2):
	hf_mid_res_prefix = f"mid_block.resnets.{j}."
	sd_mid_res_prefix = f"middle_block.{2*j}."
	unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_to_sd_checkpoint(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
	weight_err = 0.0
	converted = {}

	for k, v in state_dict.items():
		k = "lora_" + k

		for sd_prefix, hf_prefix in unet_conversion_map_layer:
			k = k.replace(hf_prefix, sd_prefix)
		
		k = k.replace('.', '_')
		k = k.replace("_lora_up_weight", ".lora_up.weight")
		k = k.replace("_lora_down_weight", ".lora_down.weight")

		converted[k] = v.to(torch.float16)
		weight_err = max(weight_err, (v.float() - converted[k].float()).abs().max().item())
	
	return converted, weight_err


################################################################################
### Main entry point
################################################################################
@hydra.main(version_base=None, config_path=None, config_name="trainer_config")
def main(config: TrainerConfig) -> None:
	"""Hydra wrapper for train."""
	if not config:
		raise ValueError("Config is empty. Please provide a valid config.")
	trainer = Trainer(config)
	return trainer.fit()


if __name__ == '__main__':
    main()