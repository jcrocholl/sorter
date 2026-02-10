import pytest
from cell import Cell


def test_col_to_int():
    assert Cell.col_to_int("A") == 1
    assert Cell.col_to_int("B") == 2
    assert Cell.col_to_int("Z") == 26


def test_int_to_col():
    assert Cell.int_to_col(1) == "A"
    assert Cell.int_to_col(2) == "B"
    assert Cell.int_to_col(26) == "Z"


def test_cell_from_string():
    cell = Cell.from_string("A1")
    assert cell.col == "A"
    assert cell.row == 1


def test_cell_from_string_larger():
    cell = Cell.from_string("G10")
    assert cell.col == "G"
    assert cell.row == 10


def test_cell_from_string_invalid():
    with pytest.raises(ValueError):
        Cell.from_string("1A")
    with pytest.raises(ValueError):
        Cell.from_string("A")
    with pytest.raises(ValueError):
        Cell.from_string("1")
