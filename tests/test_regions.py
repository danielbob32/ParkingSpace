import os
import numpy as np
from src.parkingspace.regions import save_regions_to_file, load_regions_from_file

def test_save_load_regions(tmp_path):
    # create dummy regions
    upper_level_l = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=np.int32)
    upper_level_m = np.array([[20,20],[30,20],[30,30],[20,30]], dtype=np.int32)
    ignore_regions = [np.array([[5,5],[6,5],[6,6],[5,6]], dtype=np.int32)]
    path = os.path.join(tmp_path, "test_regions.json")

    # Just re-use the same polygons for demonstration
    save_regions_to_file(
        upper_level_l,
        upper_level_m,
        upper_level_l,
        upper_level_m,
        upper_level_l,
        upper_level_m,
        upper_level_l,
        upper_level_m,
        ignore_regions,
        file_path=path
    )

    loaded = load_regions_from_file(path)
    # loaded is (upl, upm, ..., ignore)
    # For brevity, just check a couple
    assert (loaded[0] == upper_level_l).all()
    assert len(loaded[-1]) == 1
    assert (loaded[-1][0] == ignore_regions[0]).all()
