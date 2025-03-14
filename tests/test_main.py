import os
import numpy as np
import pytest

from src.parkingspace.utils import get_contour_center

def test_get_contour_center():
    # Make a simple rectangle contour
    contour = np.array([[[0,0]], [[10,0]], [[10,5]], [[0,5]]], dtype=np.int32)
    center = get_contour_center(contour)
    assert center == (5,2), f"Expected center (5,2), got {center}"
