FROM nvcr.io/nvidia/pytorch:25.05-py3

# Install dependencies
RUN pip install --no-cache-dir \
	wandb==0.18.7 \
	transformers==4.52.4 \
	diffusers==0.33.1 \
	datasets==3.6.0 \
	'mosaicml[wandb]==0.31.0' \
	mosaicml-streaming==0.12.0 \
	'huggingface_hub[hf_transfer]==0.32.3' \
	accelerate==1.7.0 \
	einops==0.8.1 \
	hydra-core==1.3.2

# Copy training script
COPY mosaic-train.py /workspace/mosaic-train.py


# Needed to avoid OOM errors
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

CMD ["composer", "mosaic-train.py", "--config-path", "./", "--config-name", "mosaic-config.yaml"]