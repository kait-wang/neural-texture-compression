from PIL import Image
import numpy as np
from utils import load_texture

def sample_bilinear(texture, u, v):
    H, W, _ = texture.shape

    # use texel center so that texels are weighted correctly during interpolation
    # convert u,v to coord sys that has texel centers at integer coords 
    x = u * W - 0.5
    y = v * H - 0.5

    # 4 surrounding texels (centers)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # cell -> connecting all four texel centers
    s = x - x0 # % across cell 
    t = y - y0 # % down cell 

    # readjust to stay within img boundaries 
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # fetch colors of the 4 texels (y,x idx for numpy imgs!)
    c00 = texture[y0c, x0c]
    c10 = texture[y0c, x1c]
    c01 = texture[y1c, x0c]
    c11 = texture[y1c, x1c]

    # bilinearly interpolate
    top = (1 - s) * c00 + s * c10
    bottom = (1 - s) * c01 + s * c11
    return (1 - t) * top + t * bottom

# test by sampling with u,v --> x,y = texel center 