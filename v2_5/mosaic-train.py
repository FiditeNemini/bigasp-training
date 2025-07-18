from collections.abc import Mapping, Sequence
import logging
import operator
import random
import time
from pathlib import Path
from typing import Any
import os

import composer.loggers.wandb_logger
import hydra
import torch
import torch.nn.functional as F
from composer import (
	Algorithm,
	Callback,
	ComposerModel,
	DataSpec,
	Evaluator,
	Logger,
	State,
	Trainer,
)
from composer.algorithms.low_precision_groupnorm import apply_low_precision_groupnorm
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from composer.devices import DeviceGPU
from composer.loggers import LoggerDestination
from composer.utils import dist, reproducibility
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
	FlowMatchEulerDiscreteScheduler,
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from omegaconf import DictConfig, OmegaConf
#from streaming import Stream, StreamingDataLoader, StreamingDataset
from flowrider import StreamingDataset, StreamingDataLoader
from flowrider import Config as StreamingConfig
from torchmetrics import MeanSquaredError
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from composer.utils import get_device
import shutil


# TODO: Is the index passed to the dataset monotonic? No.  If I want fully deterministic training I'll need to find some way to factor in the epoch in the datasets.
# NOTE: Autoresume is enabled. If it's disabled, manual resuming can be done using load_path on the Trainer's config in the config yaml.




# These tags will always be included if they are present in the tag string
IMPORTANT_TAGS = set(['watermark', 'worst quality', 'low quality', 'normal quality', 'high quality', 'best quality', 'masterpiece quality'])


def train(config: DictConfig) -> None:
	torch.set_float32_matmul_precision("high")

	try:
		total, used, free = shutil.disk_usage('/dev/shm')
		print(f"Shared memory space: {free / 1024**3:.2f} GB free, {used / 1024**3:.2f} GB used, {total / 1024**3:.2f} GB total")
	except OSError as e:
		print(f"Failed to check shared memory space: {e}")

	dist.initialize_dist(get_device(None), 300.0)

	reproducibility.seed_all(config['seed'])

	print(f"DEBUG: LOCAL_RANK: {dist.get_local_rank()}/{os.environ.get('LOCAL_RANK', 'N/A')}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")

	# Filter out dumb warnings
	#StreamingWarningFilter.setup()

	# Build model
	model = stable_diffusion_xl(base_model=config.model.base_model, base_revision=config.model.base_revision, base_variant=config.model.base_variant, train_text_encoder=config.model.train_text_encoder)
	#model.unet.enable_xformers_memory_efficient_attention()
	if config.model.gradient_checkpointing:
			model.unet.enable_gradient_checkpointing()
	
	# Compile
	# I cannot for the life of me get this to work with SDXL
	# The second compile command does work. Either I try it as is, and it fails because expandable_segments is on
	# or I use max-autotune-no-cudagraph and the result in a slower training speed
	#torch._inductor.config.conv_1x1_as_mm = True
	#torch._inductor.config.coordinate_descent_tuning = True
	#torch._inductor.config.epilogue_fusion = False
	#torch._inductor.config.coordinate_descent_check_all_directions = True
	#model.unet = torch.compile(model.unet, dynamic=True, mode='reduce-overhead', fullgraph=False)#, mode='reduce-overhead', fullgraph=False)  # type: ignore
	# model.unet = torch.compile(model.unet, mode='max-autotune', dynamic=True, fullgraph=False)  # type: ignore

	# Build optimizer
	optimizer_cls = torch.optim.AdamW
	kwargs = {
		'lr': config.optimizer.lr,
		'betas': (config.optimizer.adam_beta1, config.optimizer.adam_beta2),
		'eps': config.optimizer.adam_eps,
		'weight_decay': config.optimizer.weight_decay,
		'fused': config.optimizer.fused if 'fused' in config.optimizer else False,
	}

	base_lr = config.optimizer.lr
	param_groups: list[dict] = [
		{"params": model.unet.parameters()},  # Uses base learning rate
	]

	if config.model.train_text_encoder:
		param_groups.append(
			{"params": model.text_encoder.parameters(), "lr": base_lr * config.optimizer.te_lr_rate}
		)
	
	optimizer = optimizer_cls(param_groups, **kwargs)
	#optimized_params = list(itertools.chain.from_iterable(g['params'] for g in param_groups))

	# Build dataset streams
	train_streams = build_streams(remote=config.dataset.remote, split='train', local=config.dataset.local)
	if len(train_streams) == 0:
		raise ValueError(f"No training streams found in {config.dataset.remote} for split 'train'")
	
	test_streams = build_streams(remote=config.dataset.remote, split='test', local=config.dataset.local)
	if len(test_streams) == 0:
		raise ValueError(f"No test streams found in {config.dataset.remote} for split 'test'")
	
	stable_train_streams = build_streams(remote=config.dataset.remote, split='stable-train', local=config.dataset.local)
	if len(stable_train_streams) == 0:
		raise ValueError(f"No stable training streams found in {config.dataset.remote} for split 'stable-train'")
	
	# Build streaming datasets
	print("Building streaming datasets...")
	flowrider_config = StreamingConfig(cache_dir=config.dataset.cache_dir, cache_limit=config.dataset.cache_limit, max_downloads=config.dataset.max_downloads, readahead=config.dataset.readahead, num_cache_workers=config.dataset.num_cache_workers, trace_path="flowrider-trace.log")
	micro_batch_size = config.trainer.device_train_microbatch_size
	#test_batch_size = len(test_streams) * micro_batch_size   # TODO: Bit hacky, len(test_dataset) gets modified by num_canonical_nodes
	#stable_batch_size = len(stable_train_streams) * micro_batch_size
	train_dataset = StreamingImageCaptionDataset(streaming_config=flowrider_config, tokenizer=model.tokenizer, tokenizer_2=model.tokenizer_2, streams=train_streams, micro_batch_size=micro_batch_size, tag_prob=config.dataset.tag_prob, **config.dataset.train_dataset.streaming_kwargs)
	print(f"Train dataset has {len(train_dataset)}")
	test_dataset = StreamingImageCaptionDataset(streaming_config=flowrider_config, tokenizer=model.tokenizer, tokenizer_2=model.tokenizer_2, streams=test_streams, micro_batch_size=micro_batch_size, tag_prob=config.dataset.tag_prob, **config.dataset.test_dataset.streaming_kwargs)
	print(f"Test dataset has {len(test_dataset)}")
	stable_train_dataset = StreamingImageCaptionDataset(streaming_config=flowrider_config, tokenizer=model.tokenizer, tokenizer_2=model.tokenizer_2, streams=stable_train_streams, micro_batch_size=micro_batch_size, tag_prob=config.dataset.tag_prob, **config.dataset.test_dataset.streaming_kwargs)
	print(f"Stable train dataset has {len(stable_train_dataset)}")

	# Build straming dataloaders
	print("Building streaming dataloaders...")
	train_dataloader = CustomStreamingDataLoader(device_batch_size=config.dataset.train_batch_size // dist.get_world_size(), dataset=train_dataset, **config.dataset.train_dataset.dataloader_kwargs)
	test_dataloader = CustomStreamingDataLoader(device_batch_size=config.dataset.test_batch_size // dist.get_world_size(), dataset=test_dataset, **config.dataset.test_dataset.dataloader_kwargs)
	stable_train_dataloader = CustomStreamingDataLoader(device_batch_size=config.dataset.test_batch_size // dist.get_world_size(), dataset=stable_train_dataset, **config.dataset.test_dataset.dataloader_kwargs)

	# train_dataloader = CustomStreamingDataLoader(device_batch_size=config.dataset.train_batch_size // dist.get_world_size(), dataset=train_dataset, batch_size=micro_batch_size, sampler=None, **{**config.dataset.train_dataset.dataloader_kwargs, 'persistent_workers': False})
	# test_dataloader = CustomStreamingDataLoader(device_batch_size=test_batch_size // dist.get_world_size(), dataset=test_dataset, batch_size=micro_batch_size, sampler=None, **{**config.dataset.test_dataset.dataloader_kwargs, 'persistent_workers': False})
	# stable_train_dataloader = CustomStreamingDataLoader(device_batch_size=stable_batch_size // dist.get_world_size(), dataset=stable_train_dataset, batch_size=micro_batch_size, sampler=None, **{**config.dataset.test_dataset.dataloader_kwargs, 'persistent_workers': False})

	# # Warm up the dataloaders
	# print("Warming up test dataloader...")
	# next(iter(test_dataloader))
	# del test_dataloader
	# print("Warming up stable train dataloader...")
	# next(iter(stable_train_dataloader))
	# del stable_train_dataloader
	# print("Warming up train dataloader...")
	# next(iter(train_dataloader))
	# del train_dataloader

	# Recreate the dataloaders, since Composer wants them fresh
	# Build straming dataloaders
	#print("Building streaming dataloaders again...")
	#train_dataloader = CustomStreamingDataLoader(device_batch_size=config.dataset.train_batch_size // dist.get_world_size(), dataset=train_dataset, batch_size=micro_batch_size, sampler=None, **config.dataset.train_dataset.dataloader_kwargs)
	#test_dataloader = CustomStreamingDataLoader(device_batch_size=test_batch_size // dist.get_world_size(), dataset=test_dataset, batch_size=micro_batch_size, sampler=None, **config.dataset.test_dataset.dataloader_kwargs)
	#stable_train_dataloader = CustomStreamingDataLoader(device_batch_size=stable_batch_size // dist.get_world_size(), dataset=stable_train_dataset, batch_size=micro_batch_size, sampler=None, **config.dataset.test_dataset.dataloader_kwargs)

	# Wrap in DataSpec to override the split_batch method
	print("Wrapping dataloaders in DataSpec...")
	train_dataloader = DataSpec(train_dataloader, split_batch=CustomStreamingDataLoader.split_batch, get_num_samples_in_batch=CustomStreamingDataLoader.get_num_samples_in_batch)
	test_dataloader = DataSpec(test_dataloader, split_batch=CustomStreamingDataLoader.split_batch, get_num_samples_in_batch=CustomStreamingDataLoader.get_num_samples_in_batch)
	stable_train_dataloader = DataSpec(stable_train_dataloader, split_batch=CustomStreamingDataLoader.split_batch, get_num_samples_in_batch=CustomStreamingDataLoader.get_num_samples_in_batch)

	# Need to sleep for a bit to avoid dataloader crash
	# This is from MosaicML's diffusion repo, so I assume it is needed
	#print("Sleeping for 10 seconds to avoid dataloader crash...")
	#time.sleep(10)

	# Evaluators
	eval_set = [
		Evaluator(label='test_loss', dataloader=test_dataloader, metric_names=['MeanSquaredError']),
		Evaluator(label='stable_train_loss', dataloader=stable_train_dataloader, metric_names=['MeanSquaredError'])
	]

	# Get run name
	os.environ['WANDB_RUN_ID'] = config.run_id

	# Build list of loggers, callbacks, and algorithms to pass to trainer
	callbacks: list[Callback] = []
	algorithms: list[Algorithm] = []
	loggers: list[LoggerDestination] = []

	if 'logger' in config:
		for log, log_conf in config.logger.items():
			if '_target_' in log_conf:
				print(f'Instantiating logger <{log_conf._target_}>')
				if log == 'wandb':
					container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
					wandb_logger = hydra.utils.instantiate(log_conf, _partial_=True)
					loggers.append(wandb_logger(init_kwargs={'config': container}))
				else:
					loggers.append(hydra.utils.instantiate(log_conf))
			else:
				print(f'Logger <{log}> does not have a _target_ field, skipping instantiation.')

	#container = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
	#wandb_logger = composer.loggers.wandb_logger.WandBLogger(
	#	project=config.logger.wandb.project,
	#	init_kwargs={'config': container},
	#)
	#loggers.append(wandb_logger)
	# loggers.append(composer.loggers.FileLogger(
	# 	filename="logs/{run_name}/logs-rank{rank}.txt",
	# 	remote_file_name="s3://big-asp-v2-5-checkpoints/logs/{run_name}/logs-rank{rank}.txt",
	# 	flush_interval=50,
	# ))
	# loggers.append(composer.loggers.RemoteUploaderDownloader(
	# 	bucket_uri="s3://big-asp-v2-5-checkpoints",
	# ))

	if 'algorithms' in config:
		for ag_name, ag_conf in config.algorithms.items():
			if '_target_' in ag_conf:
				print(f'Instantiating algorithm <{ag_conf._target_}>')
				algorithms.append(hydra.utils.instantiate(ag_conf))
			elif ag_name == 'low_precision_groupnorm':
				surgery_target = model
				if 'attribute' in ag_conf:
					surgery_target = operator.attrgetter(ag_conf.attribute)(model)
				apply_low_precision_groupnorm(
					model=surgery_target,
					precision=Precision(ag_conf['precision']),
					optimizers=optimizer,
				)
			elif ag_name == 'low_precision_layernorm':
				surgery_target = model
				if 'attribute' in ag_conf:
					surgery_target = operator.attrgetter(ag_conf.attribute)(model)
				apply_low_precision_layernorm(
					model=surgery_target,
					precision=Precision(ag_conf['precision']),
					optimizers=optimizer,
				)
	
	if 'callbacks' in config:
		for call_conf in config.callbacks.values():
			if '_target_' in call_conf:
				print(f'Instantiating callbacks <{call_conf._target_}>')
				callbacks.append(hydra.utils.instantiate(call_conf))

	scheduler = hydra.utils.instantiate(config.scheduler)

	#prof = composer.profiler.Profiler(
	#	schedule = composer.profiler.cyclic_schedule(wait=0, warmup=1, active=5, repeat=1),
	#	trace_handlers = [composer.profiler.JSONTraceHandler(folder="traces")],
	#)

	print("Initializing Trainer...")
	trainer: Trainer = hydra.utils.instantiate(
		config.trainer,
		run_name=config.run_id,
		train_dataloader=train_dataloader,
		eval_dataloader=eval_set,
		optimizers=optimizer,
		model=model,
		loggers=loggers,
		algorithms=algorithms,
		schedulers=scheduler,
		callbacks=callbacks,
		#profiler=prof,
	)

	logging.info(f"Trainer initialized. GPU is {dist.get_world_size()}x{dist.get_local_rank()} on {get_device(None)}")

	print(trainer.state.model)

	# Set GradScaler growth interval
	assert trainer.state.scaler is not None, "GradScaler is not set in the trainer state. Make sure to use a GradScaler in the optimizer."
	trainer.state.scaler.set_growth_interval(500000 // config.dataset.train_batch_size)

	def eval_and_then_train():
		if config.get('eval_first', True):
			print("Running evaluation before training...")
			trainer.eval()
		print("Starting training...")
		trainer.fit()

	return eval_and_then_train()


def stable_diffusion_xl(
	base_model: str,
	base_revision: str = "main",
	base_variant: str = "fp16",
	latent_scaling_factor: float = 0.13025,
	fsdp: bool = True,
	train_text_encoder: bool = True,
) -> "StableDiffusion":
	metrics = [MeanSquaredError()]

	tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=base_revision, use_fast=True)
	tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", revision=base_revision, use_fast=True)

	text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", revision=base_revision, torch_dtype=torch.float32, variant=base_variant, use_safetensors=True)
	text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", revision=base_revision, torch_dtype=torch.float32, variant=base_variant, use_safetensors=True)

	unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", revision=base_revision, torch_dtype=torch.float32, variant=base_variant, use_safetensors=True)
	assert isinstance(unet, UNet2DConditionModel)

	if hasattr(unet, 'mid_block') and unet.mid_block is not None:
		for attention in unet.mid_block.attentions:
			attention._fsdp_wrap = True # type: ignore
		for resnet in unet.mid_block.resnets:
			resnet._fsdp_wrap = True # type: ignore
	for block in unet.up_blocks:
		if hasattr(block, 'attentions'):
			for attention in block.attentions: # type: ignore
				attention._fsdp_wrap = True
		if hasattr(block, 'resnets'):
			for resnet in block.resnets: # type: ignore
				resnet._fsdp_wrap = True
	for block in unet.down_blocks:
		if hasattr(block, 'attentions'):
			for attention in block.attentions: # type: ignore
				attention._fsdp_wrap = True
		if hasattr(block, 'resnets'):
			for resnet in block.resnets: # type: ignore
				resnet._fsdp_wrap = True

	# Make the noise schedulers
	noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    # Make the composer model
	model = StableDiffusion(
		unet=unet,
		text_encoder=text_encoder,
		text_encoder_2=text_encoder_2,
		tokenizer=tokenizer,
		tokenizer_2=tokenizer_2,
		noise_scheduler=noise_scheduler,
		latent_scaling_factor=latent_scaling_factor,
		metrics=metrics,
		fsdp=fsdp,
		train_text_encoder=train_text_encoder,
	)

	if torch.cuda.is_available():
		# Note: DeviceGPU by default enables TF32
		model = DeviceGPU().module_to_device(model)

	return model



class StableDiffusion(ComposerModel):
	def __init__(
    	self,
		unet: UNet2DConditionModel,
		text_encoder: CLIPTextModel,
		text_encoder_2: CLIPTextModelWithProjection,
		tokenizer: CLIPTokenizer,
		tokenizer_2: CLIPTokenizer,
		noise_scheduler,
		latent_scaling_factor: float,
		loss_fn=F.mse_loss,
		metrics: list | None = None,
		fsdp: bool = False,
		train_text_encoder: bool = True,
	):
		super().__init__()
		self.unet = unet
		self.noise_scheduler = noise_scheduler
		self.latent_scaling_factor = latent_scaling_factor
		self.loss_fn = loss_fn
		self.metrics = metrics if metrics is not None else [MeanSquaredError()]
		self.tokenizer = tokenizer
		self.tokenizer_2 = tokenizer_2
		self.text_encoder = text_encoder
		self.text_encoder_2 = text_encoder_2
		self.text_encoder.requires_grad_(train_text_encoder)
		self.text_encoder_2.requires_grad_(False)
		self.train_text_encoder = train_text_encoder
		if fsdp:
			# only wrap models we are training
			self.text_encoder._fsdp_wrap = train_text_encoder # type: ignore
			self.text_encoder_2._fsdp_wrap = False # type: ignore
			self.unet._fsdp_wrap = True # type: ignore

		self.rng_generator: torch.Generator | None = None

	def _generate_timesteps(self, latents: torch.Tensor):
		# Flow matching
		u = compute_density_for_timestep_sampling(
			weighting_scheme='logit_normal',
			batch_size=latents.shape[0],
			logit_mean=0.0,
			logit_std=1.0,
			mode_scale=1.29,
		)
		indices = (u * self.noise_scheduler.config.num_train_timesteps).long() # type: ignore
		timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
		return timesteps

	def set_rng_generator(self, rng_generator: torch.Generator):
		self.rng_generator = rng_generator

	def forward(self, batch):
		# for i in range(1000000):
		# 	path = Path("debug_forward") / f"forward_batch_{i}.pt"
		# 	if path.exists():
		# 		continue
		# 	path.parent.mkdir(parents=True, exist_ok=True)

		# 	torch.save({
		# 		"latent": batch['latent'],
		# 		"original_size": batch['original_size'],
		# 		"crop": batch['crop'],
		# 		"target_size": batch['target_size'],
		# 		"prompt": batch['prompt'],
		# 		"prompt_2": batch['prompt_2'],
		# 	}, path)
		# 	break

		#with open('forward_log.txt', 'a') as f:
		#	f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forwarding ({self.training=}): {batch['index'].tolist()}\n")
		#print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Forwarding batch with shapes: {batch['latent'].shape}, {batch['prompt'].shape}, {batch['prompt_2'].shape}")
		latents: torch.Tensor = batch['latent']
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
		timesteps = self._generate_timesteps(latents)

		# Add noise to the inputs (forward diffusion)
		# Noise is added according to flow matching: zt = (1 - texp) * x + texp * z1
		sigmas = get_sigmas(self.noise_scheduler, latents.device, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
		noise = torch.randn(latents.size(), device=latents.device, dtype=latents.dtype, generator=self.rng_generator)
		noised_latents = (1.0 - sigmas) * latents + sigmas * noise
		noised_latents.requires_grad_(False)

		# Generate the targets (flowmatch)
		targets = noise - latents

		if not hasattr(self, 'debug_write_training_batch') and dist.get_local_rank() == 0:
			self.debug_write_training_batch = True
			torch.save({
				"noisy_latents": noised_latents,
				"timesteps": timesteps,
				"text_embeds": text_embeds,
				"add_time_ids": add_time_ids,
				"noise": noise,
				"batch": batch,
				"text_pooled_embeds": text_pooled_embeds,
			}, "debug_training_batch.pt")

		# Forward through the model
		return self.unet(
			noised_latents,
			timesteps,
			encoder_hidden_states=text_embeds,
			added_cond_kwargs={
				"text_embeds": text_pooled_embeds,
				"time_ids": add_time_ids,
			}
		).sample, targets, timesteps

	def loss(self, outputs, batch):
		"""Loss between unet output and added noise, typically mse."""
		loss = self.loss_fn(outputs[0], outputs[1])
		if loss.isnan().any() or loss.mean() > 2.0:
			print(f"WARNING: Loss is NaN or too high: {loss.mean()}. Outputs: {outputs[0].mean()}, Targets: {outputs[1].mean()}")
			for i in range(1000):
				path = f"debug_explosion_{i}.pt"
				if not Path(path).exists():
					torch.save({
						"outputs": outputs[0],
						"targets": outputs[1],
						"batch": batch,
						"loss": loss,
					}, path)
					print(f"Saved debug data to {path}")
					break
		return loss

	def eval_forward(self, batch, outputs=None):
		"""For stable diffusion, eval forward computes unet outputs as well as some samples."""
		# Skip this if outputs have already been computed, e.g. during training
		if outputs is not None:
			return outputs
		return self.forward(batch)

	def get_metrics(self, is_train: bool = False):
		metrics_dict = {metric.__class__.__name__: metric for metric in self.metrics}
		return metrics_dict

	def update_metric(self, batch, outputs, metric):
		metric.update(outputs[0], outputs[1])


# def get_sigmas(noise_scheduler_copy, device, timesteps, n_dim=4, dtype=torch.float32):
# 	sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
# 	schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
# 	timesteps = timesteps.to(device)
# 	step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

# 	sigma = sigmas[step_indices].flatten()
# 	while len(sigma.shape) < n_dim:
# 		sigma = sigma.unsqueeze(-1)

# 	return sigma

def get_sigmas(
	noise_scheduler,
	device,
	timesteps: torch.Tensor,
	n_dim: int = 4,
	dtype: torch.dtype = torch.float32,
):
	sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype, non_blocking=True)
	sched_ts = noise_scheduler.timesteps.to(device, non_blocking=True)

	# 2. Vectorised lookup ----------------------------------------------------
	# sched_ts is descending; flip to ascending so we can use searchsorted,
	# then map the indices back to the original order.
	asc_ts = torch.flip(sched_ts, dims=(0,))            # ascending copy
	idx = torch.searchsorted(asc_ts, timesteps.to(device))
	idx = (len(sched_ts) - 1) - idx                     # restore descending offset

	sigma = sigmas.index_select(0, idx)

	# 3. Append as many trailing singleton dims as needed in one shot.
	extra = (1,) * max(n_dim - sigma.ndim, 0)
	return sigma.reshape(*sigma.shape, *extra)


class StreamingImageCaptionDataset(StreamingDataset):
	def __init__(
		self,
		tokenizer: PreTrainedTokenizer,
		tokenizer_2: PreTrainedTokenizer,
		streams: Sequence[tuple[str, str]],
		tag_prob: float,
		streaming_config: StreamingConfig,
		**streaming_kwargs,
	) -> None:
		super().__init__(
			remotes_and_locals=streams,
			config=streaming_config,
			**streaming_kwargs,
		)

		self.tokenizer = tokenizer
		self.tokenizer_2 = tokenizer_2
		self.tag_prob = tag_prob
		self.token_counts = {}

	def __getitem__(self, index):
		sample = self.get_sample(index)
		filehash: bytes = sample['filehash']
		tag_string: str = sample['tag_string']
		caption: str = sample['caption']
		score: int = int(sample['score'])
		latent_bytes: bytes = sample['latent']
		n_chunks: int = int(sample['n_chunks'])
		latent_width: int = int(sample['latent_width'])
		latent_height: int = int(sample['latent_height'])
		original_width: int = int(sample['original_width'])
		original_height: int = int(sample['original_height'])
		crop_x: int = int(sample['crop_x'])
		crop_y: int = int(sample['crop_y'])

		# Load the latent
		assert len(latent_bytes) == 4 * 4 * latent_width * latent_height, f"Expected {latent_width}x{latent_height}x4x4 bytes, got {len(latent_bytes)}, for {filehash.hex()}"
		latent = torch.frombuffer(bytearray(latent_bytes), dtype=torch.float32).view(4, latent_width, latent_height)

		# Build prompt
		if torch.rand(1) < self.tag_prob and tag_string != '':
			prompt = self.build_prompt_from_tags(tag_string, score, n_chunks)
		else:
			prompt = caption
		
		# UCG
		if torch.rand(1) < 0.05:
			if torch.rand(1) < 0.5:
				prompt = ""
			else:
				prompt = score_to_quality_string(score)

		# Tokenize the prompt
		tokens = self.tokenizer.encode(prompt, padding=False, truncation=False, add_special_tokens=False, verbose=False)
		tokens_2 = self.tokenizer_2.encode(prompt, padding=False, truncation=False, add_special_tokens=False, verbose=False)

		# Chunk the tokens to n_chunks
		# Each chunk is 75 tokens, bookended by BOS and EOS tokens (77 tokens total)
		# If any chunk is shorter than 77 tokens, it is padded with pad tokens
		tokens = chunk_tokens(tokens, n_chunks, self.tokenizer) # type: ignore
		tokens_2 = chunk_tokens(tokens_2, n_chunks, self.tokenizer_2) # type: ignore

		return {
			'latent': latent,
			'prompt': tokens,
			'prompt_2': tokens_2,
			'original_size': torch.tensor([original_height, original_width], dtype=torch.long),
			'crop': torch.tensor([crop_y, crop_x], dtype=torch.long),
			'target_size': torch.tensor([latent_width * 8, latent_height * 8], dtype=torch.long),   # goofed on height vs width; fixed by reversing here
			'index': torch.tensor(index, dtype=torch.long),
		}
	
	def get_token_count(self, s: str) -> int:
		"""Memoizes the token count for a string."""
		if s in self.token_counts:
			return self.token_counts[s]
		else:
			tokens = self.tokenizer.encode(s, add_special_tokens=False, truncation=False, padding=False)
			self.token_counts[s] = len(tokens)
			return len(tokens)

	def build_prompt_from_tags(self, tag_string: str, score: int, n_chunks: int) -> str:
		"""
		Builds a prompt from the tag string.
		A quality tag is added 90% of the time.
		Randomly chooses between using underscores, spaces, or mixed.
		The tags included in the prompt are randomly selected from the tag string, with important tags always included.
		The final prompt length will be a randomly length, but always n_chunks (NOTE: Might be slightly longer on rare occasions).
		Tags are separated by either "," or ", " (randomly).
		"""
		tags = set(tag.strip() for tag in tag_string.split(",") if tag.strip())

		# Add the quality string as a tag 90% of the time
		if torch.rand(1) > 0.1:
			tags.add(score_to_quality_string(score))
		
		# Randomly choose between underscores, spaces, or mixed
		tag_type = random.randint(0, 2)
		processed_tags = []
		important_tags = set()
		for tag in tags:
			if tag_type == 0 or (tag_type == 2 and random.random() < 0.5):
				mod_tag = tag.replace(" ", "_")
			else:
				mod_tag = tag.replace("_", " ")
			
			if tag in IMPORTANT_TAGS:
				important_tags.add(mod_tag)
			else:
				processed_tags.append(mod_tag)
		
		# Construct final set of tags
		# Important tags are always included, after that tags are added until the limit is reached
		token_limit = random.randint((n_chunks - 1) * 75, n_chunks * 75)
		token_limit = max(10, token_limit)  # Minimum 10 tokens
		final_tags = list(important_tags)
		token_count = sum(self.get_token_count(tag) for tag in final_tags)
		random.shuffle(processed_tags)

		for tag in processed_tags:
			cost = self.get_token_count(tag) + 1 # +1 for the comma, which is roughly correct

			if token_count + cost > token_limit:
				break

			final_tags.append(tag)
			token_count += cost
		
		# Build the final prompt
		prompt = ""
		random.shuffle(final_tags)

		for tag in final_tags:
			if len(prompt) > 0:
				prompt += ","
				if random.random() < 0.8:
					prompt += " "
			
			prompt += tag
		
		return prompt


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


def score_to_quality_string(score: int) -> str:
	assert score >= 0 and score <= 10, f"Score {score} is out of range [0, 10]"
	if score <= 1:
		return "worst quality"
	elif score <= 3:
		return "low quality"
	elif score <= 5:
		return "normal quality"
	elif score <= 6:
		return "high quality"
	elif score <= 8:
		return "best quality"
	else:
		return "masterpiece quality"


def build_streams(remote: str, split: str, local: str) -> list[tuple[str, str]]:
	"""
	Builds a list of streams from a remote directory by iterating all subdirectories.
	Args:
		remote (str): The remote directory where the dataset is stored.
		split (str): The split of the dataset, e.g. 'train', 'val', 'test'.
		local (Path): The local directory where the dataset will be cached.
	Returns:
		list[tuple[str, str]]: A list of tuples where each tuple contains the remote URL of the stream and the local path.
	"""
	local_path = Path(local) / split
	streams: list[tuple[str, str]] = []

	if remote.startswith("s3://"):
		if not remote.endswith('/'):
			remote += '/'
		remote += split + '/'
		stream_paths = s3_list_directories(remote)
	else:
		remote_path = Path(remote) / split
		if not remote_path.exists():
			raise ValueError(f"Remote path {remote_path} does not exist. Please check the remote directory.")
		stream_paths = [f"file://{path.absolute()}" for path in remote_path.iterdir() if path.is_dir()]
	
	for stream_path in stream_paths:
		stream_name = stream_path.split('/')[-1] if not stream_path.endswith('/') else stream_path.split('/')[-2]
		print(f"Building stream for {stream_path} in split {split}, caching to {local_path} as {stream_name}")

		stream = (stream_path, str(local_path / stream_name))
		streams.append(stream)
	
	return streams


def s3_list_directories(remote_path: str) -> list[str]:
	"""
	Return each immediate sub-directory as a fully-qualified S3/R2 URI
	like  's3://my-bucket/photos/2025/01/'.
	"""
	import os
	from urllib.parse import urlparse

	import boto3
	from botocore.config import Config

	# Split into bucket and prefix
	if remote_path.startswith("s3://"):
		u = urlparse(remote_path)
		bucket, prefix = u.netloc, u.path.lstrip("/")
	else:
		bucket, _, prefix = remote_path.partition("/")
	if prefix and not prefix.endswith("/"):
		prefix += "/"

	# List directories
	s3 = boto3.client(
		"s3",
		endpoint_url=os.getenv("S3_ENDPOINT_URL"),
		region_name="auto",
		config=Config(
			signature_version="s3v4",
			s3={"addressing_style": "path"},
		),
	)

	paginator = s3.get_paginator("list_objects_v2")
	uris: list[str] = []

	for page in paginator.paginate(
		Bucket=bucket,
		Prefix=prefix,
		Delimiter="/",
	):
		for cp in page.get("CommonPrefixes", []):
			# cp["Prefix"] is bucket-relative → turn it into a full URI
			uris.append(f"s3://{bucket}/{cp['Prefix']}")

	return uris



class GradScalerMonitor(Callback):
	def batch_end(self, state: State, logger: Logger):
		if state.scaler is None:
			print("No scaler found, skipping scale logging.")
		else:
			logger.log_metrics({'scalar': state.scaler.get_scale()})


class CustomStreamingDataLoader(StreamingDataLoader):
	"""
	A custom streaming dataloader.
	Right now, Composer is set up to do microbatches in a loop. So it asks for a whole device batch at a time, then splits it into microbatches.
	That forces the dataloader to return a whole device batch at a time, which makes life very difficult when we are trying to do different
	tensor shapes for each _microbatch_.
	This custom class allows us to set the StreamingDataset to return microbatches instead of device batches.
	They get collected into a single device batch, and then by overriding "split_batch" we can split the device batch into the original microbatches.
	"""
	def __init__(self, device_batch_size: int, **kwargs):
		super().__init__(**kwargs)
		self.device_batch_size = device_batch_size
	
	# def __len__(self):
	# 	"""
	# 	Returns the number of device batches.
	# 	This is the total number of microbatches divided by the device batch size.
	# 	"""
	# 	micro_batch_size = self.dataset.batch_size # type: ignore
	# 	micro_batches_per_device_batch = self.device_batch_size // micro_batch_size
	# 	num_micro_batches = super().__len__()
	# 	num_device_batches = num_micro_batches // micro_batches_per_device_batch
	# 	remaining = num_micro_batches % micro_batches_per_device_batch
	# 	#print(f"DEBUG: DataLoader::__len__ -> {micro_batch_size=}, {micro_batches_per_device_batch=}, {num_micro_batches=}, {num_device_batches=}, {remaining=}")
	# 	if self.drop_last or remaining == 0:
	# 		return num_device_batches
	# 	else:
	# 		return num_device_batches + 1
	
	def __iter__(self):
		current_batch = []
		current_batch_size = 0

		for batch in super().__iter__():
			current_batch.append(batch)
			current_batch_size += self.get_num_samples_in_batch(batch)

			if current_batch_size > self.device_batch_size:
				raise ValueError(f"Device batch size ({self.device_batch_size}) must be evenly divisible by the microbatch size ({self.get_num_samples_in_batch(batch)}).")

			if current_batch_size == self.device_batch_size:
				yield BatchOfMicrobatches(current_batch)
				current_batch = []
				current_batch_size = 0
		
		if len(current_batch) > 0 and not self.drop_last:
			yield BatchOfMicrobatches(current_batch)
		elif len(current_batch) > 0 and self.drop_last:
			print(f"Dropping last batch of size {current_batch_size} with {len(current_batch)} microbatches, since drop_last is set to True.")

	@staticmethod
	def split_batch(batch: Any, microbatch_size: int | float) -> Sequence:
		assert isinstance(batch, BatchOfMicrobatches), f"Expected batch to be a BatchOfMicrobatches, got {type(batch)}"
		# The batch is already a list of microbatches, so we just return it as is.
		return batch.microbatches
	
	# @staticmethod
	# def get_num_samples_in_batch(batch: Any) -> int:
	# 	# This method could be more generic, but it's good enough for us.
	# 	# Handles: list[dict[str, torch.Tensor]] and dict[str, torch.Tensor]
	# 	if isinstance(batch, dict):
	# 		assert len(batch) > 0, "Batch must not be empty"
	# 		x = next(iter(batch.values()))
	# 		assert isinstance(x, torch.Tensor), f"Expected batch item to contain a tensor, got {type(x)}"
	# 		assert x.ndim > 0, f"Expected batch item tensor to have at least one dimension, got {x.ndim}"
	# 		return x.shape[0]
	# 	elif isinstance(batch, list):
	# 		assert len(batch) > 0, "Batch must not be empty"
	# 		return sum(CustomStreamingDataLoader.get_num_samples_in_batch(b) for b in batch)
	# 	else:
	# 		raise ValueError(f"Expected batch to be a dict or a list, got {type(batch)}")
	
	@staticmethod
	def get_num_samples_in_batch(batch) -> int:
		"""
		Figure out how many *individual* samples are in `batch`.
		Handles common collate outputs (tensor, mapping, sequence).
		"""
		if isinstance(batch, BatchOfMicrobatches):
			return sum(StreamingDataLoader._infer_batch_size(mb) for mb in batch.microbatches)
		
		if torch.is_tensor(batch):
			return batch.size(0)
		
		if isinstance(batch, Mapping):
			return StreamingDataLoader._infer_batch_size(next(iter(batch.values())))
		
		if isinstance(batch, Sequence):
			return len(batch)
		
		raise  TypeError(f"Cannot infer batch size of type {type(batch)}")


class BatchOfMicrobatches:
	def __init__(self, microbatches: Sequence):
		self.microbatches = microbatches


class StreamingWarningFilter(logging.Filter):
	"""
	StreamingDataset gives a useless warning that we need to filter out, otherwise it floods the logs.
	"""
	def filter(self, record):
		return not (
			record.levelname == 'WARNING' and 
			'device_per_stream' in record.getMessage() and
			'batches with an inadequate number of samples' in record.getMessage()
		)

	@staticmethod
	def setup():
		streaming_logger = logging.getLogger('streaming.base.batching.device_per_stream')
		streaming_logger.addFilter(StreamingWarningFilter())


@hydra.main(version_base=None, config_path=".", config_name="mosaic-config")
def main(config: DictConfig) -> None:
    """Hydra wrapper for train."""
    if not config:
        raise ValueError("""\
                            Config path and name not specified!
                            Please specify these by using --config-path and --config-name, respectively.""")
    return train(config)


if __name__ == '__main__':
    main()