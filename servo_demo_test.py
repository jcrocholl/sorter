import sys
from unittest.mock import MagicMock, patch


def test_servo_demo_main_loop_reaches_all_channels(monkeypatch):
    # Create and inject mocks before importing the module under test
    mock_adafruit = MagicMock()
    mock_pca = MagicMock()
    mock_adafruit.PCA9685.return_value = mock_pca

    monkeypatch.setitem(sys.modules, "adafruit_pca9685", mock_adafruit)
    monkeypatch.setitem(sys.modules, "board", MagicMock())
    monkeypatch.setitem(sys.modules, "busio", MagicMock())

    # Now we can safely import the demo
    import servo_demo  # noqa: E402

    # Patch ServoController.send_angle to capture calls and stop the loop
    with patch("servo_controller.ServoController.send_angle") as mock_send:
        # Stop after 16 calls (one for each row)
        mock_send.side_effect = [None] * 16 + [KeyboardInterrupt()]

        try:
            servo_demo.main()
        except KeyboardInterrupt:
            pass

        # Verify it attempted to send to all 16 rows
        channels_sent = set()
        for call in mock_send.call_args_list:
            print(repr(call))
            channel = call.kwargs["channel"]
            channels_sent.add(channel)

        assert len(channels_sent) == 16
        assert channels_sent == set(range(0, 16))
        print("Demo logic verified: all 16 channels targeted.")
