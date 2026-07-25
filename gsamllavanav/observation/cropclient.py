"""
An rgbd image client that uses cropped raster image.

Examples
--------
>>> cropclient.load_image_cache()
>>> rgb, depth = cropclient.crop_view_area('birmingham_block_0', Pose4D(350, 243, 30, np.pi/4), (500, 500))
"""
import gc
from typing import Literal
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.mask
from tqdm import tqdm

from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.defaultpaths import ORTHO_IMAGE_DIR, get_data_root
from gsamllavanav.space import Pose4D, view_area_corners


# module-wise image cache
_raster_cache = None
_rgb_cache = None
_height_cache = None  # can be converted to depth

MODEL_RGB_SHAPE = (224, 224)
MODEL_DEPTH_SHAPE = (256, 256)
MODEL_OBS_DIVISIBILITY = 16


def get_rgbd(map_name: str, pose: Pose4D, rgb_size: tuple[int, int], depth_size: tuple[int, int]):
    
    rgb = crop_image(map_name, pose, rgb_size, 'rgb')
    depth = crop_image(map_name, pose, depth_size, 'depth')

    return rgb, depth


def crop_model_image(map_name: str, pose: Pose4D, type: Literal['rgb', 'depth']) -> np.ndarray:
    target_shape = MODEL_RGB_SHAPE if type == 'rgb' else MODEL_DEPTH_SHAPE
    image = crop_image(map_name, pose, target_shape, type)
    image = _normalize_fixed_obs_shape(image, target_shape, type)
    _validate_model_obs_shape(target_shape)
    return image


def crop_image(map_name: str, pose: Pose4D, shape: tuple[int, int], type: Literal['rgb', 'depth']) -> np.ndarray:

    image = (_rgb_cache if type =='rgb' else _height_cache)[map_name]
    
    view_area_corners_rowcol = _compute_view_area_corners_rowcol(map_name, pose)
    view_area_corners_colrow = np.flip(view_area_corners_rowcol, axis=-1)

    img_row, img_col = shape
    img_corners_colrow = [(0, 0), (img_col-1, 0), (img_col - 1, img_row - 1), (0, img_row - 1)]
    img_corners_colrow = np.array(img_corners_colrow, dtype=np.float32)
    img_transform = cv2.getPerspectiveTransform(view_area_corners_colrow, img_corners_colrow)
    cropped_image = cv2.warpPerspective(image, img_transform, shape)

    if type == 'depth':
        cropped_image = pose.z - cropped_image
        cropped_image = cropped_image[..., np.newaxis]

    return cropped_image


def _normalize_fixed_obs_shape(
    image: np.ndarray,
    target_shape: tuple[int, int],
    type: Literal['rgb', 'depth'],
) -> np.ndarray:
    if image.shape[:2] == target_shape:
        return image

    interpolation = cv2.INTER_LINEAR if type == 'rgb' else cv2.INTER_NEAREST
    resized = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=interpolation)
    if type == 'depth' and resized.ndim == 2:
        resized = resized[..., np.newaxis]
    return resized


def _validate_model_obs_shape(shape: tuple[int, int]) -> None:
    if shape[0] % MODEL_OBS_DIVISIBILITY != 0 or shape[1] % MODEL_OBS_DIVISIBILITY != 0:
        raise ValueError(
            f"Model obs shape {shape} must be divisible by {MODEL_OBS_DIVISIBILITY}"
        )


def _compute_view_area_corners_rowcol(map_name: str, pose: Pose4D):
    """Returns the [front-left, front-right, back-right, back-left] corners of
    the view area in (row, col) order
    """

    raster = _raster_cache[map_name]

    view_area_corners_rowcol = [raster.index(x, y) for x, y in view_area_corners(pose, GROUND_LEVEL[map_name])]

    return np.array(view_area_corners_rowcol, dtype=np.float32)


def load_image_cache(image_dir=None, alt_env: Literal['', 'flood', 'ground_fissure'] = ''):
    if image_dir is None:
        image_dir = get_data_root() / "rgbd"
    image_dir = Path(image_dir)
    if alt_env:
        image_dir = image_dir/alt_env

    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is None:
        _raster_cache = {
            raster_path.stem: rasterio.open(raster_path)
            for raster_path in image_dir.glob("*.tif")
        }

    if _rgb_cache is None:
        _rgb_cache = {
            rgb_path.stem: cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            for rgb_path in tqdm(image_dir.glob("*.png"), desc="reading rgb data from disk", leave=False)
        }

    if _height_cache is None:
        _height_cache = {
            map_name: raster.read(1)  # read first channel (1-based index)
            for map_name, raster in tqdm(_raster_cache.items(), desc="reading depth data from disk", leave=False)
        }



def clear_image_cache():
    global _raster_cache, _rgb_cache, _height_cache

    if _raster_cache is not None:
        for dataset in _raster_cache:
            dataset.close()
    
    _raster_cache = _rgb_cache = _height_cache = None
    gc.collect()
