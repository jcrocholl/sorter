import servo_channel
from unittest.mock import MagicMock
import pytest


def test_col_to_int():
    assert servo_channel.col_to_int("A") == 1
    assert servo_channel.col_to_int("B") == 2
    assert servo_channel.col_to_int("Z") == 26


def test_int_to_col():
    assert servo_channel.int_to_col(1) == "A"
    assert servo_channel.int_to_col(2) == "B"
    assert servo_channel.int_to_col(26) == "Z"


def test_send_angle():
    controller = MagicMock()
    servo = servo_channel.ServoChannel(col="B", row=7, controller=controller, channel=6)
    servo.send_angle(135)
    controller.send_angle.assert_called_once_with(channel=6, angle=135)


def test_parse_range():
    assert servo_channel.parse_range("A1") == [("A", 1)]
    assert servo_channel.parse_range("A1:A5") == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("A", 4),
        ("A", 5),
    ]


def test_parse_range_invalid():
    with pytest.raises(ValueError):
        servo_channel.parse_range("1A")
    with pytest.raises(ValueError):
        servo_channel.parse_range("A")
    with pytest.raises(ValueError):
        servo_channel.parse_range("1")
    with pytest.raises(ValueError):
        servo_channel.parse_range("A1:10")


def test_parse_ranges():
    controller = MagicMock(frequency=50.0)  # Hz
    servos = servo_channel.parse_ranges("A1:A10 B5", controller)
    assert len(servos) == 11
    assert " ".join(str(c) for c in servos) == "A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 B5"

    assert servos[0].col == "A"
    assert servos[0].row == 1
    assert servos[0].controller == controller
    assert servos[0].channel == 0

    assert servos[-1].col == "B"
    assert servos[-1].row == 5
    assert servos[-1].controller == controller
    assert servos[-1].channel == 10
