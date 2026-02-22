import sys
from unittest.mock import MagicMock, patch
import pytest

# Create and inject mocks before importing the module under test
mock_adafruit = MagicMock()
mock_pca = MagicMock()
mock_adafruit.PCA9685.return_value = mock_pca

sys.modules["adafruit_pca9685"] = mock_adafruit
sys.modules["board"] = MagicMock()
sys.modules["busio"] = MagicMock()

# Now we can safely import the demo
import servo_demo  # noqa: E402


def test_servo_demo_main_loop_reaches_all_channels():
    # Patch ServoController.send_angle to capture calls and stop the loop
    with patch("servo_demo.ServoController.send_angle") as mock_send:
        # Stop after 16 calls (one for each row)
        mock_send.side_effect = [None] * 16 + [KeyboardInterrupt()]

        try:
            servo_demo.main()
        except KeyboardInterrupt:
            pass

        # Verify it attempted to send to all 16 rows
        rows_sent = set()
        for call in mock_send.call_args_list:
            cell = call[0][0]
            rows_sent.add(cell.row)

        assert len(rows_sent) == 16
        assert rows_sent == set(range(1, 17))
        print("Demo logic verified: all 16 rows targeted.")


if __name__ == "__main__":
    pytest.main([__file__])
