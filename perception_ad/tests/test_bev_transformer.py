import numpy as np
import cv2
import pytest
from pathlib import Path
from bev_transformer import BEVTransformer
import yaml

 # load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

@pytest.fixture
def bev_transformer():
    return BEVTransformer(config=cfg)

def test_batch_warp_folder(tmp_path: Path, bev_transformer: BEVTransformer):
    """
    Reads all PNGs in tests/data/image_data,
    warps each into BEV, writes to tmp_path/bev_images,
    and asserts the outputs exist and have expected shape.
    """
    # locate your “checked-in” test data folder:
    repo_root      = Path(__file__).parent.parent
    image_data_dir = repo_root / "tests" / "data" / "image_data"
    assert image_data_dir.exists(), f"No test images in {image_data_dir}"

    out_dir = repo_root / "tests" / "data" / "bev_images"
    out_dir.mkdir(exist_ok=True)

    # process each image
    # 3) warp and save
    for img_file in image_data_dir.glob("*.png"):
        img = cv2.imread(str(img_file))
        bev = bev_transformer.warp(img)
        out_file = out_dir / img_file.name
        cv2.imwrite(str(out_file), bev)

        # 4) assertions
        assert out_file.exists()
        res = cv2.imread(str(out_file))
        # Note: dst_size is (width, height)
        assert res.shape[1] == bev_transformer.dst_size[0]
        assert res.shape[0] == bev_transformer.dst_size[1]