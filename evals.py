import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from train import train, reconstruct
from quantization import quantize_model, quantized_model_size
from utils import * 


# all 9 texture-architecture configs

TEXTURES = {
    # "gradient": "textures/gradient.png",
    # "bricks": "textures/bricks.png",
    # "clouds": "textures/clouds.png",
    "water": "textures/water.png",
    "dirt": "textures/dirt.png",
    "ships": "textures/ships.png"
}

ARCHITECTURES = {
    "Small": {
        "resolutions": (64,),
        "feat_dim": 2
    },

    "Medium": {
        "resolutions": (16, 32, 64),
        "feat_dim": 2
    },

    "Large": {
        "resolutions": (16, 32, 64, 128),
        "feat_dim": 4
    }
}

OUTPUT_DIR = "nn_outputs"


# helper for calculating compression ratio 
def model_size(model):
    """
    Float32 neural representation size.
    Every learned parameter (MLP + feature grids) is stored as 4 bytes.
    """

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    num_bytes = num_params * 4
    return num_params, num_bytes


def raw_texture_size(texture):
    """
    Raw RGB8 texture -->  H * W * 3 bytes
    """

    H, W, _ = texture.shape
    return H * W * 3

# original vs reconstructed comparisons 
def save_comparison(
    original,
    recon,
    texture_name,
    architecture_name,
    final_psnr,
    neural_kb,
    raw_kb,
    compression_ratio,
    path
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{raw_kb:.2f} KB")
    axes[0].axis("off")

    axes[1].imshow(recon)
    axes[1].set_title(
        f"{architecture_name}\n"
        f"{final_psnr:.2f} dB | "
        f"{neural_kb:.2f} KB"
    )
    axes[1].axis("off")

    fig.suptitle(
        f"{texture_name.capitalize()} — "
        f"{architecture_name}\n"
        f"compressed/original = "
        f"{compression_ratio:.4f}"
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


# plot psnr over training
def plot_training_curves(histories):
    '''
    get PSNR training curve for all 9 configs 
    '''

    plt.figure(figsize=(11, 7))

    for history in histories:

        label = (
            f"{history['texture']} - "
            f"{history['architecture']}"
        )

        plt.plot(
            history["steps"],
            history["psnrs"],
            label=label
        )

    plt.xlabel("Training Step")
    plt.ylabel("Full-image PSNR (dB)")
    plt.title("Neural Texture Compression Training")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "all_training_curves.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def save_results_csv(results):
    path = os.path.join(
        OUTPUT_DIR,
        "p6_results.csv"
    )
    fields = [
        "texture",
        "architecture",
        "resolutions",
        "feat_dim",
        "psnr",
        "num_params",
        "neural_kb",
        "raw_kb",
        "compression_ratio"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


# ===========================================================================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()

    results = []
    histories = []
    total_runs = len(TEXTURES) * len(ARCHITECTURES)
    run_number = 0

    # test quantization 
    quantization_results = []

    # run all 9 configs 
    for texture_name, texture_path in TEXTURES.items():
        for architecture_name, config in ARCHITECTURES.items():
            run_number += 1

            print("\n" + "=" * 70)
            print(
                f"RUN {run_number}/{total_runs}: "
                f"{texture_name} / "
                f"{architecture_name}"
            )

            print("=" * 70)

            # one model per config 
            (
                model,
                losses,
                eval_steps,
                eval_psnrs,
                texture
            ) = train(
                texture_path=texture_path,
                resolutions=config["resolutions"],
                feat_dim=config["feat_dim"],
                steps=2000,
                batch_size=16384,
                lr=1e-2,
                device=device,
            )

            # get reconstruction and final psnr
            recon, final_psnr = reconstruct(model, texture, device)

            # get compression ratio
            num_params, neural_bytes = model_size(model)
            raw_bytes = raw_texture_size(texture)
            neural_kb = neural_bytes / 1024
            raw_kb = raw_bytes / 1024

            compression_ratio = neural_bytes / raw_bytes

            # save comparison images 

            comparison_path = os.path.join(
                OUTPUT_DIR,
                f"{texture_name}_"
                f"{architecture_name.lower()}_"
                f"comparison.png"
            )

            save_comparison(
                texture,
                recon,
                texture_name,
                architecture_name,
                final_psnr,
                neural_kb,
                raw_kb,
                compression_ratio,
                comparison_path
            )

            # save PSNR throughout training for plotting later 
            histories.append({
                "texture": texture_name,
                "architecture":
                    architecture_name,
                "steps": eval_steps,
                "psnrs": eval_psnrs
            })

            results.append({
                "texture": texture_name,
                "architecture": architecture_name,
                "resolutions": str(config["resolutions"]),
                "feat_dim": config["feat_dim"],
                "psnr": final_psnr,
                "num_params": num_params,
                "neural_kb": neural_kb,
                "raw_kb": raw_kb,
                "compression_ratio": compression_ratio
            })

            # print results
            print("\nFINAL RESULT")
            print(f"PSNR: {final_psnr:.2f} dB")
            print(f"Parameters: {num_params:,}")
            print(f"Neural size: {neural_kb:.2f} KB")
            print(f"Raw size: {raw_kb:.2f} KB")
            print(f"Compressed / Original: {compression_ratio:.4f}")


            # ==============================================================================
            # quantization after training
            # ==============================================================================

            psnr_before = final_psnr
            grid_bytes = quantize_model(model) # quantize feature grids 

            # evaluate quantized model for lost quality 
            quant_recon, psnr_after = reconstruct(model, texture, device)
            psnr_loss = psnr_before - psnr_after 

            # calculate new size and compression ration 
            quantized_bytes = quantized_model_size(model, grid_bytes)

            quantized_kb = quantized_bytes / 1024
            quantized_ratio = quantized_bytes / raw_bytes

            quantization_results.append({
                "texture": texture_name,
                "architecture": architecture_name,
                "psnr_before": psnr_before,
                "psnr_after": psnr_after,
                "psnr_loss": psnr_loss,
                "size_kb": quantized_kb,
                "compression_ratio": quantized_ratio
            })


            print("\nP7:")
            print(f"PSNR before: {psnr_before:.2f}")
            print(f"PSNR after:  {psnr_after:.2f}")
            print(f"PSNR loss:   {psnr_loss:.2f}")
            print(f"New size:    {quantized_kb:.2f} KB")
            print(f"New ratio:   {quantized_ratio:.4f}")

    # all 9 models evaluated 
    save_results_csv(results)
    plot_training_curves(histories)

    print("\n" + "=" * 80)
    print("P6 FINAL RESULTS")
    print("=" * 80)

    print(
        f"{'Texture':<12}"
        f"{'Model':<10}"
        f"{'PSNR':>12}"
        f"{'Size KB':>14}"
        f"{'Raw KB':>14}"
        f"{'Ratio':>12}"
    )

    for r in results:
        print(
            f"{r['texture']:<12}"
            f"{r['architecture']:<10}"
            f"{r['psnr']:>12.2f}"
            f"{r['neural_kb']:>14.2f}"
            f"{r['raw_kb']:>14.2f}"
            f"{r['compression_ratio']:>12.4f}"
        )

    print(
        f"\nOutputs saved to "
        f"{OUTPUT_DIR}/"
    )

    # print quantization results  
    print("\n" + "=" * 80)
    print("P7 QUANTIZATION RESULTS")
    print("=" * 85)

    print(
        f"{'Texture':<12}"
        f"{'Model':<10}"
        f"{'Before':>10}"
        f"{'After':>10}"
        f"{'Loss':>10}"
        f"{'Size KB':>12}"
        f"{'Ratio':>10}"
    )

    for r in quantization_results:
        print(
            f"{r['texture']:<12}"
            f"{r['architecture']:<10}"
            f"{r['psnr_before']:>10.2f}"
            f"{r['psnr_after']:>10.2f}"
            f"{r['psnr_loss']:>10.2f}"
            f"{r['size_kb']:>12.2f}"
            f"{r['compression_ratio']:>10.4f}"
        )

if __name__ == "__main__":
    main()