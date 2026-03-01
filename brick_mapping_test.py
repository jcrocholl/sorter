import pytest
from brick_mapping import BrickMapping


@pytest.fixture
def brick_mapping():
    # Using the real file since it's available and small
    return BrickMapping("drawers/brick_classes.csv")


def test_brick_mapping_get_cell(brick_mapping):
    # Based on the content of drawers/brick_classes.csv
    # Line 1: 3007_brick_2x8,3034_plate_2x8...
    assert brick_mapping.get_cell("3007_brick_2x8") == "A1"
    assert brick_mapping.get_cell("3034_plate_2x8") == "B1"

    # Line 3: 3001_brick_2x4...
    assert brick_mapping.get_cell("3001_brick_2x4") == "A3"

    # Line 10: 3009_brick_1x6...
    assert brick_mapping.get_cell("3009_brick_1x6") == "A10"


def test_brick_mapping_get_class(brick_mapping):
    assert brick_mapping.get_class("A1") == "3007_brick_2x8"
    assert brick_mapping.get_class("B1") == "3034_plate_2x8"
    assert brick_mapping.get_class("A3") == "3001_brick_2x4"
    assert brick_mapping.get_class("G10") == "11213_plate_round_6x6"


def test_brick_mapping_not_found(brick_mapping):
    with pytest.raises(KeyError):
        brick_mapping.get_cell("unknown_class")

    with pytest.raises(KeyError):
        brick_mapping.get_class("Z99")
