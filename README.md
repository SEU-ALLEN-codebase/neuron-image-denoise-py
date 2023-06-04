# Python Package for Neuron Image Denoising
The 3D neuron image denoising filter cythonized.

# Install
```bash
$ pip install neuron-image-denoise
```

# Guide

```python
import neuron_image_denoise as nid

img: numpy.ndarray # 3D numpy array, 16bit

out = nid.adaptive_denoise(img, ada_interval=(2, 3, 3), flare_interval=(2, 8, 8),
                           ada_sampling=3, flare_sampling=8, flare_weight=.02,
                           atten_depth=4, flare_x=True, flare_y=True)

out = nid.adaptive_denoise_16to8(img, ada_interval=(2, 3, 3), flare_interval=(2, 8, 8),
                           ada_sampling=3, flare_sampling=8, flare_weight=.02,
                           atten_depth=4, flare_x=True, flare_y=True)   # 16bit to 8bit
```