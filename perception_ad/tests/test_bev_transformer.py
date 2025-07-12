import numpy as np
import cv2
import pytest
from pathlib import Path
from bev_transformer import BEVTransformer

def make_checkerboard(w, h, grid=20):
    # create a simple black/white checker so we can test warp↔unwarp
    img = np.indices((h, w)).sum(axis=0) // grid % 2 * 255
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def test_warp_and_unwarp_identity():
    w, h = 200, 100
    # full-image trapezoid to rectangle
    src = np.float32([[0,h],[w,h],[w,0],[0,0]])
    bev = BEVTransformer(src_pts=src, dst_size=(w,h))
    img = make_checkerboard(w, h)
    warped   = bev.warp(img)
    unwarped = bev.unwarp(warped)
    # unwarped should match original (within interpolation error)
    assert unwarped.shape == img.shape
    diff = np.abs(unwarped.astype(int) - img.astype(int))
    assert np.mean(diff) < 2  # tolerance for bilinear filtering

@pytest.fixture
def bev_transformer():
    # adjust these source points to match your camera’s ROI
    # for tests we assume images are 1280×720
    width, height = 1280, 720
    src_pts = np.float32([
        [0, height],      # bottom-left
        [width, height],    # bottom-right
        [width, height*0.5],  # right horizon
        [0, height*0.5]   # left horizon
    ])
    bev_size = (640, 360)
    return BEVTransformer(src_pts=src_pts, dst_size=bev_size)

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
    out_dir.mkdir()

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