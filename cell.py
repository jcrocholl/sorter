from dataclasses import dataclass
import re

_CELL_REGEX = re.compile(r"^([A-Z]+)(\d+)$")


@dataclass
class Cell:
    """Address for a spreadsheet cell such as A1 or G10."""

    col: str  # e.g. A to Z
    row: int  # e.g. 1 to 10

    @property
    def col_int(self) -> int:
        return self.col_to_int(self.col)

    @staticmethod
    def from_string(text: str) -> "Cell":
        """Parse A1 into Cell('A', 1)."""
        m = _CELL_REGEX.match(text)
        if m is None:
            raise ValueError(text)
        return Cell(
            col=m.group(1),
            row=int(m.group(2)),
        )

    @staticmethod
    def col_to_int(col: str) -> int:
        """Convert column str 'A' to int 1."""
        assert "A" <= col <= "Z"
        return 1 + ord(col) - ord("A")

    @staticmethod
    def int_to_col(i: int) -> str:
        """Convert int 1 to column str 'A'."""
        assert 1 <= i <= 26
        return chr(ord("A") + i - 1)
