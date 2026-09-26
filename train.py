import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils import *

from nn_texture import NeuralTexture

# no train/test split - trying to overfit to image! 
def build_dataset(texture, device):
    """
    X: texel-center coordinate
    y: RGB target

    texture: numpy array (H, W, 3)

    returns:
        coords: (H*W, 2)
        target: (H*W, 3)
    """

    H, W, _ = texture.shape

    # make texel center coords across entire image
    u = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    v = (torch.arange(H, dtype=torch.float32) + 0.5) / H

    # Create all possible texel locations
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    coords = torch.stack([uu, vv], dim=-1)
    coords = coords.reshape(-1, 2) # (H, W, 2) -> (H*W, 2)

    # RGB targets:
    target = torch.from_numpy(texture).reshape(-1, 3) # (H, W, 2) -> (H*W, 2)

    return coords.to(device), target.to(device)


def train(
    texture_path,
    resolutions=(16, 32, 64, 128),
    feat_dim=2,
    steps=2000,
    batch_size=16384,
    lr=1e-2,
    device=None
):

    """
    Fit one NeuralTexture model to one image.
    Returns: model, losses, psnrs, texture
    """

    if device is None:
        device = get_device()

    # load texture img

    texture = load_texture(texture_path)
    H, W, _ = texture.shape

    print(f"Texture: {texture_path}")
    print(f"Resolution: {W} x {H}")

    # make dataset
    coords, target = build_dataset(texture, device)
    num_texels = coords.shape[0]
    print(f"Number of texels: {num_texels}")

    #initialize model and Adam optimizer
    model = NeuralTexture(
        resolutions=resolutions,
        feat_dim=feat_dim
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr 
    )

    # train over batch of 16384 randomly sampled texels for 2000 steps 
    losses = []
    eval_steps = []
    psnrs = []


    model.train()
    for step in range(steps):

        idx = torch.randint(
            0,
            num_texels,
            (batch_size,), # 16384
            device=device
        )

        batch_coords = coords[idx]
        batch_target = target[idx]

        pred = model(batch_coords)

        loss = F.mse_loss(pred, batch_target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        # per-batch psnr 
        # psnr = -10.0 * torch.log10(loss)

        # psnr over full image using current model state every 50 steps to make plot for P6
        if step % 50 == 0 or step == steps - 1:
            full_psnr = evaluate_psnr(model, coords, target)

            eval_steps.append(step)
            psnrs.append(full_psnr)

            print(
                f"Step {step:4d}/{steps} | "
                f"Batch loss: {loss.item():.6f} | "
                f"Full PSNR: {full_psnr:.2f} dB"
            )

    return model, losses, eval_steps, psnrs, texture

 
@torch.no_grad()
def reconstruct(model, texture, device, batch_size=65536):
    '''
    build img back up using model predictions
    '''
    H, W, _ = texture.shape
    coords, target = build_dataset(texture, device)
    model.eval()
    predictions = []

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        pred = model(batch_coords)
        predictions.append(pred.cpu())

    predictions = torch.cat(predictions, dim=0)
    recon = predictions.reshape(H, W, 3).numpy()

    mse = np.mean((recon - texture) ** 2)
    final_psnr = -10 * np.log10(mse)

    return recon, final_psnr

@torch.no_grad()
def evaluate_psnr(model, coords, target, batch_size=65536):
    '''
    helper function for train() to get psnr over entire img instead of a sample batch for P6 plot

    same as reconstruct() but takes in coords/targets and doesn't save preds 
    '''
    model.eval()

    squared_error_sum = 0.0
    num_values = 0

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        batch_target = target[start:start + batch_size]

        pred = model(batch_coords)

        squared_error_sum += torch.sum(
            (pred - batch_target) ** 2
        ).item()

        num_values += batch_target.numel()

    mse = squared_error_sum / num_values
    psnr = -10 * np.log10(mse)

    model.train()
    return psnr

# test on gradient texture 
def main():
    device = get_device()

    model, losses, psnrs, texture = train(
        texture_path="textures/gradient.png",
        resolutions=(16, 32, 64),
        feat_dim=2,
        steps=2000,
        batch_size=16384,
        lr=1e-2,
        device=device
    )

    recon, final_psnr = reconstruct(
        model,
        texture,
        device
    )

    print(f"\nWhole-image PSNR: {final_psnr:.2f} dB")
    os.makedirs("nn_outputs", exist_ok=True)

    recon_uint8 = (
        np.clip(recon, 0, 1) * 255
    ).astype(np.uint8)

    Image.fromarray(recon_uint8).save(
        "nn_outputs/gradient_neural_test.png"
    )


if __name__ == "__main__":
    main()
