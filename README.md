# Neuron Image Denoising
The 3D neuron image denoising filter cythonized.

# Install
```bash
$ pip install neuron-image-denoise
```

# User Guide

```python
from neuron_image_denoise.filter import *

img: numpy.ndarray  # 3D numpy array, 16bit

params = {
    'ada_interval': (2, 3, 3),
    'flare_interval': (2, 8, 8),
    'ada_sampling': 3,
    'flare_sampling': 8,
    'flare_weight': .02,
    'atten_depth': 4,
    'flare_x': True,
    'flare_y': True
}

out = adaptive_denoise(img, **params)

out = adaptive_denoise_16to8(img, **params)  # 16bit to 8bit

out = adaptive_sectional_feedforward_filter(img, sigma=12., truncate=2., scaling=1, suppression=.8)
```

## Algorithm Explanation

Neurons are illuminated by the fluorescent proteins, which can be seen as dots in the space. Measuring the signal is to 
locate where the dots are, but the problem is, these dots can diffuse to outside of neurons and their light can be
spreaded far away due to refraction and microcopy aberration.

Adaptive thresholding can reduce this effect by assuming that true signals stand out from the background, as the
fluorescent given by the dots attenuates with distance.
Methods like difference of gaussian play similarly, but adaptive threshold has the best and most stable effect 
together with way lower computation.

Yet there's another problem caused by the microscopy, and varies depending on the instrument. The images can be polluted by
nearby slices, resulting in flares that look like a brush head. It is especially common with strong signals like somata,
making the nearby neurites discernible. There are two ways to solve this, either using more advanced microscopy to
reduce this effect to a small range, or measure this effect and cancel computationally.

# Useful Links

Github project: https://github.com/SEU-ALLEN-codebase/neuron-image-denoise-py

Documentation: https://SEU-ALLEN-codebase.github.io/neuron-image-denoise-py