import csv
from servo_channel import int_to_col


class BrickMapping:
    """Maps brick class names to shelf drawer cell names (e.g., A1)."""

    def __init__(self, csv_path: str) -> None:
        self.cell_to_class: dict[str, str] = {}
        self.class_to_cell: dict[str, str] = {}

        with open(csv_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row_idx, row in enumerate(reader, start=1):
                for col_idx, brick_class in enumerate(row, start=1):
                    brick_class = brick_class.strip()
                    if not brick_class:
                        continue

                    col_str = int_to_col(col_idx)
                    cell_label = f"{col_str}{row_idx}"
                    self.cell_to_class[cell_label] = brick_class
                    self.class_to_cell[brick_class] = cell_label

    def get_cell(self, brick_class: str) -> str:
        """Get the cell label (e.g., 'A1') for a given brick class."""
        return self.class_to_cell[brick_class]

    def get_class(self, cell_label: str) -> str:
        """Get the brick class name for a given cell label."""
        return self.cell_to_class[cell_label]
