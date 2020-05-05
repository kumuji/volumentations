[![Tests](https://github.com/kumuji/volumentations/workflows/Tests/badge.svg)](https://github.com/kumuji/volumentations/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/kumuji/volumentations/branch/master/graph/badge.svg)](https://codecov.io/gh/kumuji/volumentations)
[![PyPI](https://img.shields.io/pypi/v/volumentations.svg)](https://pypi.org/project/volumentations/)
[![Documentation Status](https://readthedocs.org/projects/volumentations/badge/?version=latest)](https://volumentations.readthedocs.io/en/latest/?badge=latest)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Downloads](https://pepy.tech/badge/volumentations)](https://pepy.tech/project/volumentations)

# Volumentations

Python library for 3d data augmentaiton. Hard fork from [alumentations](https://github.com/albumentations-team/albumentations).

For more information on available augmentations check [documentation](https://volumentations.readthedocs.io/en/latest/index.html).

Or, check simple example in colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CT9nIGME_M4kIDc3BfEF4pCb_8JdFLpH)

# Setup

`pip install volumentations`

# Usage example

```python
import volumentations as V
import numpy as np

volume_aug = V.Compose(
    [
        V.Scale3d(scale_limit=[0.1, 0.1, 0.1], bias=[1, 1, 1]),
        V.RotateAroundAxis3d(axis=[0, 0, 1], rotation_limit=np.pi / 6),
        V.RotateAroundAxis3d(axis=[0, 1, 0], rotation_limit=np.pi / 6),
        V.RotateAroundAxis3d(axis=[1, 0, 0], rotation_limit=np.pi / 6),
        V.RandomDropout3d(dropout_ratio=0.2),
    ]
)
original_point_cloud = np.empty((1000, 3))
augmented_point_cloud = volume_aug(points=original_point_cloud)["points"]

```

```
# So far the package in WIP stage
```
