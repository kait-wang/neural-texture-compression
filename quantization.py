import torch

def quantize_uint8(x):
    '''
    quantizes one array (grid)
    '''
    lo = x.min()
    hi = x.max()

    # Avoid divide-by-zero for constant arrays
    if torch.isclose(lo, hi):
        scale = torch.tensor(1.0, device=x.device)
        q = torch.zeros_like(x, dtype=torch.uint8)
    else:
        scale = (hi - lo) / 255
        q = torch.round((x - lo) / scale)
        q = q.clamp(0, 255).to(torch.uint8)

    # Decode back to float for rendering --> this is where quality might be lost
    # x -> q -> x_hat
    x_hat = lo + q.float() * scale

    return q, lo, scale, x_hat


def quantize_model(model):
    """
    quantize all feature grids only (MLP is already tiny anyways)

    returns number of bytes needed to store quantized grids
    """

    grid_bytes = 0

    with torch.no_grad():
        for grid in model.grid.grids:
            q, lo, scale, x_hat = quantize_uint8(grid.data)
            # replace grids with x_hat for this assignment but usually q is what's stored in the model 
            grid.data.copy_(x_hat)
            # count bytes 
            grid_bytes += q.numel()
            grid_bytes += 8

    return grid_bytes


def quantized_model_size(model, grid_bytes):
    """
    Now grids = uint8, MLP = float32
    """

    mlp_params = sum(
        p.numel()
        for p in model.mlp.parameters()
    )

    mlp_bytes = mlp_params * 4
    return grid_bytes + mlp_bytes