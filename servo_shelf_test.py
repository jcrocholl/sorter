import time
from unittest.mock import MagicMock

import pytest

from servo_shelf import ServoShelf


def pca_factory(addr):
    return MagicMock()


def test_servo_shelf_init():
    # Example configuration
    config = {
        0x41: "A1:A10 B1:B5",
        0x42: "B6:B10 C1:C10",
    }

    pcas = {
        0x41: MagicMock(name="pca_0x41"),
        0x42: MagicMock(name="pca_0x42"),
    }

    shelf = ServoShelf(config, lambda addr: pcas[addr])

    # Check total number of servos:
    # 0x41: A1..A10 (10) + B1..B5 (5) = 15
    # 0x42: B6..B10 (5) + C1..C10 (10) = 15
    # Total: 30
    assert len(shelf.servos) == 30

    # Check specific servos on controller 0x41
    a1 = shelf.servos["A1"]
    assert a1.col == "A"
    assert a1.row == 1
    assert a1.controller.pca == pcas[0x41]
    assert a1.channel == 0

    a10 = shelf.servos["A10"]
    assert a10.row == 10
    assert a10.controller.pca == pcas[0x41]
    assert a10.channel == 9

    b1 = shelf.servos["B1"]
    assert b1.col == "B"
    assert b1.row == 1
    assert b1.controller.pca == pcas[0x41]
    assert b1.channel == 10

    b5 = shelf.servos["B5"]
    assert b5.row == 5
    assert b5.controller.pca == pcas[0x41]
    assert b5.channel == 14

    # Check specific servos on controller 0x42
    b6 = shelf.servos["B6"]
    assert b6.col == "B"
    assert b6.row == 6
    assert b6.controller.pca == pcas[0x42]
    assert b6.channel == 0

    b10 = shelf.servos["B10"]
    assert b10.row == 10
    assert b10.controller.pca == pcas[0x42]
    assert b10.channel == 4

    c1 = shelf.servos["C1"]
    assert c1.col == "C"
    assert c1.row == 1
    assert c1.controller.pca == pcas[0x42]
    assert c1.channel == 5

    c10 = shelf.servos["C10"]
    assert c10.row == 10
    assert c10.controller.pca == pcas[0x42]
    assert c10.channel == 14


def test_servo_shelf_missing_servo():
    config = {0x41: "A1:A5"}

    shelf = ServoShelf(config, pca_factory)

    with pytest.raises(KeyError):
        _ = shelf.servos["Z1"]


def test_servo_shelf_collision():
    # Duplicate range A1:A5 across two controllers
    config = {
        0x41: "A1:A5",
        0x42: "A1:A5",
    }

    with pytest.raises(AssertionError, match="Duplicate servo label: A1"):
        ServoShelf(config, pca_factory)


def test_servo_shelf_add_event_sorting():
    config = {0x41: "A1:A5"}
    shelf = ServoShelf(config, pca_factory)

    # Add events out of order
    shelf.add_event(100.0, "A1", 90.0)
    shelf.add_event(50.0, "A2", 45.0)
    shelf.add_event(150.0, "A3", 0.0)
    shelf.add_event(75.0, "A4", 180.0)

    # Check that the queue is sorted by timestamp
    expected = [
        (50.0, "A2", 45.0),
        (75.0, "A4", 180.0),
        (100.0, "A1", 90.0),
        (150.0, "A3", 0.0),
    ]
    assert shelf._queue == expected


def test_servo_shelf_process_queue():
    pca = MagicMock()
    config = {0x41: "A1:A5"}

    shelf = ServoShelf(config, lambda addr: pca)

    # Plan events in the future and past
    now = time.time()
    shelf.add_event(now - 1.0, "A1", 90.0)  # Past
    shelf.add_event(now + 0.1, "A2", 45.0)  # Near future

    # Check initial state: no calls yet
    assert pca.pwm_regs.__setitem__.call_count == 0

    shelf.start()
    try:
        # Wait for "past" event to be processed
        # (It should be picked up almost immediately)
        time.sleep(0.05)
        # Check that A1 (channel 0) was called
        # ServoController.send_pwm_regs sets pca.pwm_regs[channel] = (on, off)
        # We check if __setitem__ was called with index 0.
        assert any(
            call.args[0] == 0 for call in pca.pwm_regs.__setitem__.call_args_list
        ), "A1 (channel 0) should have been processed"

        # Now wait for the near future event
        time.sleep(0.1)
        assert any(
            call.args[0] == 1 for call in pca.pwm_regs.__setitem__.call_args_list
        ), "A2 (channel 1) should have been processed"
    finally:
        shelf.stop()


def test_servo_shelf_stop_thread():
    config = {0x41: "A1:A5"}
    shelf = ServoShelf(config, pca_factory)

    shelf.start()
    assert shelf._thread is not None
    assert shelf._thread.is_alive()

    shelf.stop()
    assert shelf._thread is None
