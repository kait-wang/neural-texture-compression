import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

# 3 --> 2 bytes per color 
def quantize_rgb565(c):
    levels = np.array([31, 63, 31], dtype=np.float32)

    q = np.round(c * levels)
    c_hat = q / levels

    return c_hat

# 48 bytes --> 8 bytes per block  
def compress_block(block):
    pixels = block.reshape(-1, 3) # flatten from (4, 4, 3) to (16, 3)

    # PCA to get endpoints 
    mean = pixels.mean(axis=0)
    centered = pixels - mean

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0] # eigenvector w largest eigenvalue (variance)

    projections = centered @ axis

    # endpoints = extermes along the block’s principal color axis
    c0 = pixels[np.argmin(projections)]
    c1 = pixels[np.argmax(projections)]
    c0 = quantize_rgb565(c0)
    c1 = quantize_rgb565(c1)

    c2 = (2 * c0 + c1) / 3
    c3 = (c0 + 2 * c1) / 3

    palette = np.stack([c0, c1, c2, c3]) # the only 4 colors a pixel in this block can be after decomp 

    # for each og pixel, find nearest palette color to replace it 
    distances = np.sum(
        (pixels[:, None, :] - palette[None, :, :]) ** 2,
        axis=2
    )

    indices = np.argmin(distances, axis=1) # elem is 0-3 to indicate best palette color for corresponding idx 
    return c0, c1, indices


def decompress_block(c0, c1, indices):
    c2 = (2 * c0 + c1) / 3
    c3 = (c0 + 2 * c1) / 3

    palette = np.stack([c0, c1, c2, c3])
    return palette[indices].reshape(4, 4, 3)


def compress_s3tc(texture):
    H, W, _ = texture.shape

    blocks = []
    for y in range(0, H, 4):
        row = []
        for x in range(0, W, 4):
            block = texture[y:y+4, x:x+4]
            c0, c1, indices = compress_block(block)
            row.append((c0, c1, indices))

        blocks.append(row)
    return blocks


def decompress_s3tc(blocks, H, W):
    recon = np.zeros((H, W, 3), dtype=np.float32)
    for by, row in enumerate(blocks):
        for bx, (c0, c1, indices) in enumerate(row):
            block = decompress_block(c0, c1, indices)

            y = by * 4
            x = bx * 4

            recon[y:y+4, x:x+4] = block
    return recon

######################################################

def save_comparison(original, recon, name, psnr_value, compression_ratio, path):
    """
    Save original and S3TC reconstruction side-by-side for writeup.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon)
    axes[1].set_title(
        f"S3TC Reconstruction\n"
        f"PSNR: {psnr_value:.2f} dB | Size ratio: {compression_ratio:.4f}"
    )
    axes[1].axis("off")

    fig.suptitle(name.capitalize())
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    texture_dir = "textures"
    output_dir = "s3tc_outputs"

    os.makedirs(output_dir, exist_ok=True)

    texture_names = [
        "gradient",
        "bricks",
        "clouds"
    ]

    results = []

    for name in texture_names:
        path = os.path.join(texture_dir, f"{name}.png")

        print(f"\nProcessing {name}...")

     
        texture = load_texture(path)
        H, W, _ = texture.shape

        blocks = compress_s3tc(texture)
        recon = decompress_s3tc(blocks, H, W)

        # -------------------------
        # psnr + compression ratio
        # -------------------------
        psnr_value = psnr(texture, recon)

        # Original: RGB8 = 3 bytes per texel
        original_bytes = H * W * 3

        # S3TC: 8 bytes per 4x4 block
        num_blocks = (H // 4) * (W // 4)
        compressed_bytes = num_blocks * 8

        compression_ratio = compressed_bytes / original_bytes

        # -------------------------
        # Save reconstructed img
        # -------------------------
        recon_path = os.path.join(
            output_dir,
            f"{name}_s3tc.png"
        )
        save_image(recon, recon_path)

        
        comparison_path = os.path.join(
            output_dir,
            f"{name}_comparison.png"
        )

        save_comparison(
            texture,
            recon,
            name,
            psnr_value,
            compression_ratio,
            comparison_path
        )

        # Save numbers
        results.append({
            "texture": name,
            "width": W,
            "height": H,
            "psnr": psnr_value,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": compression_ratio,
        })

        print(f"PSNR:              {psnr_value:.2f} dB")
        print(f"Original size:     {original_bytes / 1024:.2f} KB")
        print(f"S3TC size:         {compressed_bytes / 1024:.2f} KB")
        print(f"Size ratio:        {compression_ratio:.4f}")

    # -------------------------
    # write result metrics to txt file
    # -------------------------
    results_path = os.path.join(
        output_dir,
        "s3tc_results.txt"
    )

    with open(results_path, "w") as f:
        f.write("S3TC RESULTS\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"Texture: {r['texture']}\n")
            f.write(
                f"Resolution: {r['width']} x {r['height']}\n"
            )
            f.write(
                f"PSNR: {r['psnr']:.2f} dB\n"
            )
            f.write(
                f"Original size: "
                f"{r['original_bytes'] / 1024:.2f} KB\n"
            )
            f.write(
                f"S3TC size: "
                f"{r['compressed_bytes'] / 1024:.2f} KB\n"
            )
            f.write(
                f"Compressed/original ratio: "
                f"{r['compression_ratio']:.4f}\n"
            )
            f.write("\n")

    # -------------------------
    # Console summary table
    # -------------------------
    print("\n" + "=" * 70)
    print("FINAL S3TC RESULTS")
    print("=" * 70)

    print(
        f"{'Texture':<12}"
        f"{'PSNR (dB)':<14}"
        f"{'Original KB':<15}"
        f"{'S3TC KB':<12}"
        f"{'Ratio':<10}"
    )

    for r in results:
        print(
            f"{r['texture']:<12}"
            f"{r['psnr']:<14.2f}"
            f"{r['original_bytes']/1024:<15.2f}"
            f"{r['compressed_bytes']/1024:<12.2f}"
            f"{r['compression_ratio']:<10.4f}"
        )

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()