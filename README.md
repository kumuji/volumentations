[![Tests](https://github.com/kumuji/volumentations/workflows/Tests/badge.svg)](https://github.com/kumuji/volumentations/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/kumuji/volumentations/branch/master/graph/badge.svg)](https://codecov.io/gh/kumuji/volumentations)
[![PyPI](https://img.shields.io/pypi/v/volumentations.svg)](https://pypi.org/project/volumentations/)
[![Documentation Status](https://readthedocs.org/projects/volumentations/badge/?version=latest)](https://volumentations.readthedocs.io/en/latest/?badge=latest)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Downloads](https://pepy.tech/badge/volumentations)](https://pepy.tech/project/volumentations)
[![CodeFactor](https://www.codefactor.io/repository/github/kumuji/volumentations/badge)](https://www.codefactor.io/repository/github/kumuji/volumentations)
[![Maintainability](https://api.codeclimate.com/v1/badges/a3dc1e079290f508bf6f/maintainability)](https://codeclimate.com/github/kumuji/volumentations/maintainability)


# ![logo](./docs/logo.png "logo") Volumentations

![augmented_teapot](./docs/augmented_teapot.png "teapot")


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

augmentation = V.Compose(
    [
        V.Scale3d(scale_limit=(0.2, 0.2, 0.1), p=0.75),
        V.OneOrOther(
            V.Compose(
                [
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi, axis=(0, 0, 1), always_apply=True
                    ),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 3, axis=(0, 1, 0), always_apply=True
                    ),
                    V.RotateAroundAxis3d(
                        rotation_limit=np.pi / 3, axis=(1, 0, 0), always_apply=True
                    ),
                ],
                p=1,
            ),
            V.Flip3d(axis=(0, 0, 1)),
        ),
        V.OneOf(
            [
                V.RandomDropout3d(dropout_ratio=0.2, p=0.75),
                V.RandomDropout3d(dropout_ratio=0.3, p=0.5),
            ]
        ),
    ]
)

augmented_teapot = augmentation(points=teapot.copy())["points"]
show_augmentation(teapot, augmented_teapot)
```
