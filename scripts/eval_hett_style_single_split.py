from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsamllavanav.observation import cropclient
from scripts import eval_single_split


def _hett_crop_model_image(map_name, pose, image_type):
    if image_type == "rgb":
        return cropclient.crop_image(map_name, pose, (224, 224), "rgb")
    if image_type == "depth":
        return cropclient.crop_image(map_name, pose, (256, 256), "depth")
    raise ValueError(f"Unsupported HETT-style model crop type: {image_type}")


def main():
    cropclient.crop_model_image = _hett_crop_model_image
    eval_single_split.main()


if __name__ == "__main__":
    main()
