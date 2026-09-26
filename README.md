# Neural Texture Compression

Implementation of a neural texture compression pipeline using multi-resolution learnable feature grids and an MLP decoder. The project also includes a traditional S3TC/DXT1 baseline and post-training 8-bit quantization.

## Files

- `texture_sampler.py` — Implements bilinear texture sampling for normalized `(u, v)` coordinates (P1).
- `s3tc.py` — Implements S3TC/DXT1 block compression and decompression using RGB565 endpoints and a four-color palette (P2).
- `nn_texture.py` — Defines the multi-resolution learnable feature grids and MLP color decoder (P3–P4).
- `train.py` — Contains texture loading, dataset construction, model training, reconstruction, and PSNR evaluation (P5).
- `evals.py` — Runs the Small, Medium, and Large neural texture experiments, evaluates their reconstruction quality and storage cost, and performs post-training quantization (P6–P8).
- `quantization.py` — Implements post-training 8-bit uniform quantization of the learned feature grids (P7).
- `textures/` — Contains the provided and self-sourced texture images.
- `nn_outputs/` — Contains generated reconstructions, plots, and experiment results for MLP + feature grids approach.
- `s3tc_outputs/` — Contains generated reconstructions, plots, and experiment results for S3TC baseline. 


## Reproducing Results

Install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pillow matplotlib

```

Choose which textures to evaluate by commenting/uncommenting entries in `TEXTURES` inside `eval.py`:

```python
TEXTURES = {
    # "gradient": "textures/gradient.png",
    # "bricks": "textures/bricks.png",
    # "clouds": "textures/clouds.png",
    "water": "textures/water.png",
    "dirt": "textures/dirt.png",
    "ships": "textures/ships.png"
}
```

Then run:

```bash
python eval.py
```

Each uncommented texture is independently trained using all three architectures:

- **Small:** 64-resolution grid, feature dimension 2
- **Medium:** 16/32/64-resolution grids, feature dimension 2
- **Large:** 16/32/64/128-resolution grids, feature dimension 4

## Evaluation Output

During training, `eval.py` prints the full-image PSNR at regular training intervals.

After all experiments finish, it prints a summary for each texture and architecture containing:

- Final reconstruction PSNR
- Neural representation size in KB
- Original texture size in KB
- Compressed/original size ratio

It also evaluates post-training 8-bit quantization and reports the PSNR before and after quantization, PSNR loss, quantized model size, and new compression ratio.

Generated reconstruction comparisons and evaluation plots are saved to the `outputs/` directory.

