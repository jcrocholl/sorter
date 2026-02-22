#!/usr/bin/env python3

"""Send updates to 16 servo channels, for testing PWM hardware and cables."""

from cell import Cell
from servo_controller import ServoController


def main():
    # Use 1 column and 16 rows to test all channels on one PCA9685.
    # This corresponds to PCA9685 address 0x41 and channels 0-15.
    num_cols = 1
    num_rows = 16

    print(
        f"Initializing ServoController with {num_cols} columns and {num_rows} rows..."
    )
    sc = ServoController(num_columns=num_cols, num_rows=num_rows)

    step = 5
    # Sequence of angles: 90 -> 180, then 0 -> 90
    angles = list(range(90, 180 + step, step))
    angles += list(range(0, 90 + step, step))

    print("Starting servo loop. Press Ctrl+C to stop.")
    offset = 0
    try:
        while True:
            offset += 1
            for col_idx in range(1, num_cols + 1):
                col = Cell.int_to_col(col_idx)
                for row in range(1, num_rows + 1):
                    cell = Cell(col, row)

                    # Stagger the angles for each servo to distribute current draw.
                    # idx calculation matches channel distribution logic.
                    idx = (col_idx - 1) * num_rows + (row - 1)
                    angle = angles[(offset + idx * 4) % len(angles)]

                    # Print debug info for the first cell
                    if col == "A" and row == 1:
                        print(f"Cell {cell} angle={angle:3d}", end="\r")

                    sc.send_angle(cell, angle)
    except KeyboardInterrupt:
        print("\nStopping servo demo...")
    finally:
        # Properly de-initialize all PCA9685 instances.
        print("De-initializing controllers...")
        for col, pca in sc.column_controllers.items():
            try:
                pca.deinit()
            except Exception as e:
                print(f"Error de-initializing column {col}: {e}")


if __name__ == "__main__":
    main()
