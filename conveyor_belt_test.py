import pytest
import typeguard
from conveyor_belt import ConveyorBelt


def test_conveyor_belt_init():
    belt = ConveyorBelt(length=1000)
    assert belt.length == 1000
    assert belt.speed == 0.0


def test_conveyor_belt_calibration_single_mark():
    belt = ConveyorBelt(length=1000)
    # First sighting doesn't give speed
    belt.observed_mark_at("mark1", 0.0)
    assert belt.speed == 0.0

    # Second sighting gives speed
    belt.observed_mark_at("mark1", 2.0)
    # length 1000 mm / 2.0 s = 500 mm/s
    assert belt.speed == 500.0


def test_conveyor_belt_calibration_multiple_marks():
    belt = ConveyorBelt(length=1000)
    belt.observed_mark_at("mark1", 0.0)
    belt.observed_mark_at("mark2", 0.5)

    belt.observed_mark_at("mark1", 2.0)  # Interval 2.0s
    assert belt.speed == 500.0

    belt.observed_mark_at("mark2", 2.5)  # Interval 2.0s
    # Average interval is (2.0 + 2.0) / 2 = 2.0s
    assert belt.speed == 500.0


def test_conveyor_belt_speed_averaging():
    belt = ConveyorBelt(length=1000)
    belt.observed_mark_at("m", 0.0)
    belt.observed_mark_at("m", 2.0)  # speed 500
    belt.observed_mark_at("m", 3.0)  # interval 1.0, speed 1000
    # Average interval = (2.0 + 1.0) / 2 = 1.5
    # Speed = 1000 / 1.5 = 666.66...
    assert belt.speed == pytest.approx(666.666, rel=1e-3)


def test_conveyor_belt_predict():
    belt = ConveyorBelt(length=1000)
    belt.observed_mark_at("m", 0.0)
    belt.observed_mark_at("m", 2.0)  # speed 500 mm/s

    # Predict travel of 250mm
    # travel_time = 250 / 500 = 0.5s
    assert belt.predict_travel_time(250) == 0.5


def test_conveyor_belt_predict_no_speed():
    belt = ConveyorBelt(length=1000)
    # No calibration yet
    assert belt.predict_travel_time(100) == 0.0


def test_typeguard_violation():
    belt = ConveyorBelt(length=1000)
    with pytest.raises(typeguard.TypeCheckError):
        belt.predict_travel_time("100")  # pyrefly: ignore


def test_conveyor_belt_max_intervals():
    belt = ConveyorBelt(length=1000)
    belt.observed_mark_at("m", 0)
    for i in range(1, 10):
        belt.observed_mark_at("m", i)
    # It should only keep the last 5 intervals: (5-4, 6-5, 7-6, 8-7, 9-8)
    # Wait, calibrate appends interval if mark already exists.
    # Marks: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # Intervals: 1, 1, 1, 1, 1, 1, 1, 1, 1 (total 9)
    # After popping, it should have 5 intervals, all = 1.0
    assert len(belt._intervals) == 5
    assert belt.speed == 1000.0
