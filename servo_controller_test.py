import sys
from unittest.mock import MagicMock

# Create and inject mocks before importing the module under test
mock_adafruit = MagicMock()
mock_pca = MagicMock()
mock_adafruit.PCA9685.return_value = mock_pca

sys.modules["adafruit_pca9685"] = mock_adafruit
sys.modules["board"] = MagicMock()
sys.modules["busio"] = MagicMock()

# Now we can safely import the controller
from servo_controller import ServoController  # noqa: E402
from cell import Cell  # noqa: E402


def test_servo_controller_init():
    # Verify it creates the right number of PCA9685 instances
    sc = ServoController(num_columns=2, num_rows=10)

    # Should have created 2 controllers (for col A and B)
    assert mock_adafruit.PCA9685.call_count == 2
    # Verify address calculation (0x40 + 1, 0x40 + 2)
    mock_adafruit.PCA9685.assert_any_call(sc.i2c, address=0x41)
    mock_adafruit.PCA9685.assert_any_call(sc.i2c, address=0x42)


def test_servo_controller_send_angle():
    sc = ServoController(num_columns=2, num_rows=10)
    cell = Cell("B", 7)
    sc.send_angle(cell, 90)

    # Cell B is column 2 (address 0x42), which is sc.column_controllers["B"]
    # Verify correct channel update on the correct PCA9685 instance
    pca_b = sc.column_controllers["B"]

    # Row 7 is channel 7 + 5 = 12 (since row > 5)
    channel = 12

    pca_b.pwm_regs.__setitem__.assert_called_once()
    args, _ = pca_b.pwm_regs.__setitem__.call_args
    assert args[0] == channel
    assert isinstance(args[1], tuple)
    assert args[1] == (2047, 2354)
