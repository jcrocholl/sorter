from collections import defaultdict


class ConveyorBelt:
    """Tracks the speed of a conveyor belt and predicts object arrival times."""

    def __init__(self, length: float) -> None:
        """
        Initialize the conveyor belt.

        Args:
            length: Total belt length in millimeters (mm).
        """
        self.length = length
        self._marks = defaultdict(list)
        self._intervals = []
        self._max_intervals = 5

    def observed_mark_at(self, mark: str, timestamp: float) -> None:
        """
        Record a sighting of a permanent mark on the belt at a specific time.

        Args:
            mark: Name of the recognizer class for the mark.
            timestamp: Time (seconds) when the mark passed the camera mid-line.
        """
        marks = self._marks[mark]
        if marks:
            # Calculate interval between subsequent sightings of the same mark.
            # One interval represents one full rotation of the belt.
            interval = timestamp - marks[-1]
            if interval > 0:
                self._intervals.append(interval)
                # Keep only the most recent intervals.
                if len(self._intervals) > self._max_intervals:
                    self._intervals.pop(0)

        marks.append(timestamp)
        # We only really need the last timestamp for each mark to calculate the next interval.
        if len(marks) > 2:
            self._marks[mark] = marks[-2:]

    @property
    def speed(self) -> float:
        """
        Returns the current belt speed in millimeters per second (mm/s).

        Returns:
            Current speed, or 0.0 if not enough calibration data is available.
        """
        if not self._intervals:
            return 0.0

        avg_interval = sum(self._intervals) / len(self._intervals)
        if avg_interval <= 0:
            return 0.0

        return self.length / avg_interval

    def predict_travel_time(self, distance: float) -> float:
        """
        Predict the travel time for an object to cover a certain distance.

        Args:
            distance: Distance to travel in millimeters (mm).

        Returns:
            Travel time in seconds (s). Returns 0.0 if not enough calibration data.
        """
        current_speed = self.speed
        if current_speed <= 0:
            return 0.0

        return distance / current_speed
