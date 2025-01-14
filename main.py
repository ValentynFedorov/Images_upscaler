import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model_path = 'RealESRGAN_x4plus.pth'

# Load the model
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state_dict, strict=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize upsampler
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=False  # Use False for CPU
)

# Load and process image
try:
    img = Image.open('picture.jpg').convert('RGB')
    img = np.array(img)
    print(f"Input image shape: {img.shape}, dtype: {img.dtype}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Enhance image
try:
    output, _ = upsampler.enhance(img, outscale=4)
    if output is not None:
        output_img = Image.fromarray(output)
        output_img.save('output.jpg')
        print("Output saved as 'output.jpg'")
    else:
        print("Enhancement failed; no output generated.")
except Exception as e:
    print(f"Error during enhancement: {e}")
