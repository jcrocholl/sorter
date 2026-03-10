import pytest
import typeguard
from conveyor_belt import ConveyorBelt


def test_conveyor_belt_init():
    belt = ConveyorBelt(length=1000)
    assert belt.length == 1000
    assert belt.speed == 0.0
    assert belt.get_kicker_distance("A0") == 0.0


def test_conveyor_belt_kicker_distances():
    kicker_distances = {"A0": 440.0, "B0": 480.0}
    belt = ConveyorBelt(length=1000, kicker_distances=kicker_distances)
    assert belt.get_kicker_distance("A0") == 440.0
    assert belt.get_kicker_distance("B0") == 480.0
    assert belt.get_kicker_distance("C0") == 0.0


def test_conveyor_belt_calibration_min_intervals():
    belt = ConveyorBelt(length=1000, min_intervals=2)
    belt.observed_mark_at("mark1", 0.0)
    belt.observed_mark_at("mark1", 2.0)  # 1 interval
    assert belt.speed == 0.0

    belt.observed_mark_at("mark1", 4.0)  # 2 intervals
    # length 1000 mm / 2.0 s (avg) = 500 mm/s
    assert belt.speed == 500.0


def test_conveyor_belt_calibration_single_mark():
    # Set min_intervals=1 to match previous test logic
    belt = ConveyorBelt(length=1000, min_intervals=1)
    # First sighting doesn't give speed
    belt.observed_mark_at("mark1", 0.0)
    assert belt.speed == 0.0

    # Second sighting gives speed (1 interval)
    belt.observed_mark_at("mark1", 2.0)
    # length 1000 mm / 2.0 s = 500 mm/s
    assert belt.speed == 500.0


def test_conveyor_belt_calibration_multiple_marks():
    # Use min_intervals=1 to keep logic simple
    belt = ConveyorBelt(length=1000, min_intervals=1)
    belt.observed_mark_at("mark1", 0.0)
    belt.observed_mark_at("mark2", 0.5)

    belt.observed_mark_at("mark1", 2.0)  # Interval 2.0s
    assert belt.speed == 500.0

    belt.observed_mark_at("mark2", 2.5)  # Interval 2.0s
    # Average interval is (2.0 + 2.0) / 2 = 2.0s
    assert belt.speed == 500.0


def test_conveyor_belt_speed_averaging():
    belt = ConveyorBelt(length=1000, min_intervals=2)
    belt.observed_mark_at("m", 0.0)
    belt.observed_mark_at("m", 2.0)  # 1 interval, speed 0 (below min)
    assert belt.speed == 0.0

    belt.observed_mark_at("m", 3.6)  # 2nd interval (1.6). Avg = 1.8
    # Speed = 1000 / 1.8 = 555.55...
    assert belt.speed == pytest.approx(555.555, rel=1e-3)


def test_conveyor_belt_predict_travel_time():
    belt = ConveyorBelt(length=1000, min_intervals=1)
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
    belt = ConveyorBelt(length=1000, max_intervals=5)
    belt.observed_mark_at("m", 0)
    for i in range(1, 10):
        belt.observed_mark_at("m", i)
    # Marks: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # Intervals: 1 (0-1), 1 (1-2), ... 1 (8-9) -> Total 9 intervals
    # After pruning with while loop in observed_mark_at: should be 5 intervals
    assert len(belt._intervals) == 5
    assert belt.speed == 1000.0


def test_conveyor_belt_outlier_rejection():
    # length 1000, min_intervals 3
    belt = ConveyorBelt(length=1000, min_intervals=3)
    belt.observed_mark_at("m", 0)
    belt.observed_mark_at("m", 1)  # Interval 1.0
    belt.observed_mark_at("m", 2)  # Interval 1.0
    belt.observed_mark_at("m", 3)  # Interval 1.0 (median 1.0)
    assert belt.speed == 1000.0

    # Add an outlier (0.7s is < 3/4 * median)
    belt.observed_mark_at("m", 3.7)  # Interval 0.7
    assert belt.speed == 1000.0

    # Add another outlier (1.4s is > 4/3 * median)
    belt.observed_mark_at("m", 5.1)  # Interval 1.4
    assert belt.speed == 1000.0
