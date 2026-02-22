import adafruit_pca9685
import board
import busio
from cell import Cell


class ServoController:
    """Controls servos for a grid of cells."""

    def __init__(self, num_columns, num_rows) -> None:
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.frequency = 50  # Hz
        self.period = 1_000_000 / self.frequency  # microsec

        self.i2c = busio.I2C(
            board.SCL,  # ty: ignore[possibly-missing-attribute]
            board.SDA,  # ty: ignore[possibly-missing-attribute]
        )

        self.column_controllers: dict[str, adafruit_pca9685.PCA9685] = {}
        for i in range(1, num_columns + 1):
            col = Cell.int_to_col(i)
            pca = adafruit_pca9685.PCA9685(self.i2c, address=0x40 + i)
            pca.frequency = self.frequency
            self.column_controllers[col] = pca

        self.row_channels: dict[int, int] = {}
        for row in range(1, num_rows + 1):
            # Row goes from 1 to num_rows, channel goes 0 to 15
            self.row_channels[row] = row - 1

    def send_angle(self, cell: Cell, angle: float) -> None:
        """Sends a new angle to the servo for this cell."""
        # Find the I2C device and channel for this cell.
        pca = self.column_controllers[cell.col]
        channel = self.row_channels[cell.row]
        # Calculate pulse width for the new servo angle.
        microsec = 600 + 1800 * angle / 180
        # Distribute rising edges across the duty cycle.
        on = int(0xFFF * channel / 16) % 0xFFF
        # Tested on 2026-02-22: it's acceptable for the off time to wrap
        # around into the next cycle, as long as 0<=off<0xFFF.
        off = int(on + 0xFFF * microsec / self.period) % 0xFFF
        # Send new 12-bit values for on/off time over I2C.
        pca.pwm_regs[channel] = (on, off)

    def open(self, cell: Cell) -> None:
        """Opens the servo flap for this cell."""
        self.send_angle(cell, 135)

    def close(self, cell: Cell) -> None:
        """Closes the servo flap for this cell."""
        self.send_angle(cell, 90)
