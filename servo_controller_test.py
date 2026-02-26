from unittest.mock import MagicMock
from servo_controller import ServoController


def test_servo_controller_init():
    pca = MagicMock()
    sc = ServoController(pca, frequency=60.0)
    assert sc.num_channels == 16
    assert pca.frequency == 60.0


def test_servo_controller_send_angle():
    pca = MagicMock()
    sc = ServoController(pca)
    sc.send_angle(channel=6, angle=45)
    MagicMock.assert_called_once_with(
        pca.pwm_regs.__setitem__,
        6,
        (1535, 1749),
    )
