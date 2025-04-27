from omegaconf import OmegaConf

from src.data import (
    create_patches_fps_knn,
    normalize_pointcloud,
    TransformsCompose,
    resample_pointcloud_farthest,
    random_rotate_pointcloud,
)


def get_transforms(config: OmegaConf, train: bool) -> TransformsCompose:
    transforms = [
        (resample_pointcloud_farthest, {"num_points": config.model.num_points}, 1.0),
        (normalize_pointcloud, {}, 1.0),
    ]

    if train:
        # Augmentations
        transforms.append(
            (random_rotate_pointcloud, {
                    "axis": "z",
                    "angle_range": (-15, 15),
                    "has_colors": config.dataset.has_colors
                },
                1.0
            ),
        )

    if config.model.name == "Pointy":
        transforms.append((
            create_patches_fps_knn, {
                "n_patches": config.model.num_patches,
                "k": config.model.num_points_per_patch
            },
            1.0
        ))

    return TransformsCompose(transforms)
