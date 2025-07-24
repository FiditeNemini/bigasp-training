# bigASP v2.5 LoRA Training Scripts

A Work in Progress script for training LoRA models for the bigASP v2.5 model.  NOT user friendly; not fully tested.

## Example Usage

Put your images into a directory with captions alongside them.  For example:

```
lora-test-1/
├── 00001.png
├── 00001.txt
├── 00002.png
├── 00002.txt
├── 00003.png
├── 00003.txt
```

The filenames are arbitrary, and the images can be PNG, JPEG, or WEBP.

Run with something like:
```bash
python train-lora.py base_model=../v25checkpoints/bigaspv25-20250716.safetensors dataset_dir=lora-test-1 total_samples=3000
```

Where `base_model` is the path to the v2.5 checkpoint.  `dataset_dir` is the path to the directory containing your images and captions.  Any configuration options in the script (under `TrainerConfig`) can be overridden using the same syntax.  In the example above, `total_samples` is overridden to 3000.

The script will automatically cache latents, train the LoRA model, and save the result in the output directory (defaults to `./checkpoints/`) in a format compatible with (at least) ComfyUI.  Should also be compatible with most other tools, but just not tested yet.

I do not know optimal settings yet, and the defaults do not reflect a working configuration.