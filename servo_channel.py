from dataclasses import dataclass
import re
from servo_controller import ServoController

_CELL_REGEX = re.compile(r"^([A-Z])(\d+)$")
_RANGE_REGEX = re.compile(r"^([A-Z])(\d+):([A-Z])(\d+)$")


def col_to_int(col: str) -> int:
    """Convert column str 'A' to int 1."""
    assert "A" <= col <= "Z"
    return 1 + ord(col) - ord("A")


def int_to_col(i: int) -> str:
    """Convert int 1 to column str 'A'."""
    assert 1 <= i <= 26
    return chr(ord("A") + i - 1)


@dataclass
class ServoChannel:
    """Address for a single servo channel such as A1 or G10."""

    col: str  # A to Z
    row: int  # 1 to 10
    controller: ServoController
    channel: int  # 0 to 15

    def __str__(self) -> str:
        return self.col + str(self.row)

    def send_angle(self, angle: float) -> None:
        self.controller.send_angle(
            channel=self.channel,
            angle=angle,
        )


def parse_range(text: str) -> list[tuple[str, int]]:
    """Parse a single range into a list of col/row tuples."""
    m = _CELL_REGEX.match(text)
    if m:
        col = m.group(1)
        row = int(m.group(2))
        return [(col, row)]

    m = _RANGE_REGEX.match(text)
    if m:
        col = m.group(1)
        row1 = int(m.group(2))
        col2 = m.group(3)
        row2 = int(m.group(4))
        assert col == col2, "multi-column ranges are not supported"
        return [(col, row) for row in range(row1, row2 + 1)]

    raise ValueError(f"failed to parse range {text} (expected something like A1:A10)")


def parse_ranges(text: str, controller: ServoController) -> list[ServoChannel]:
    """Parse ranges into a list and sequentially assign servo channels."""
    results: list[ServoChannel] = []
    channel = 0
    for part in text.split():
        for col, row in parse_range(part):
            results.append(ServoChannel(col, row, controller, channel))
            channel += 1
    return results
