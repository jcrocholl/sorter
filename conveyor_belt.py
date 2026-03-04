import statistics


class ConveyorBelt:
    """Tracks the speed of a conveyor belt and predicts object arrival times."""

    def __init__(
        self,
        length: float,
        min_intervals: int = 3,
        max_intervals: int = 10,
    ) -> None:
        """
        Initialize the conveyor belt.

        Args:
            length: Total belt length in millimeters (mm).
            min_intervals: Minimum number of intervals to make predictions.
            max_intervals: Maximum number of intervals to keep for calibration.
        """
        self.length = length
        self._min_intervals = min_intervals
        self._max_intervals = max_intervals
        self._last_seen: dict[str, float] = {}
        self._intervals: list[float] = []

    def observed_mark_at(self, mark: str, timestamp: float) -> None:
        """
        Record a sighting of a permanent mark on the belt at a specific time.

        Args:
            mark: Name of the recognizer class for the mark.
            timestamp: Time (seconds) when the mark passed the camera mid-line.
        """
        last_timestamp = self._last_seen.get(mark)
        if last_timestamp is not None:
            # Calculate interval between subsequent sightings of the same mark.
            # One interval represents one full rotation of the belt.
            interval = timestamp - last_timestamp
            if interval > 0:
                self._intervals.append(interval)
                # Keep only the most recent intervals.
                while len(self._intervals) > self._max_intervals:
                    self._intervals.pop(0)

        self._last_seen[mark] = timestamp

    @property
    def speed(self) -> float:
        """
        Returns the current belt speed in millimeters per second (mm/s).

        Discards obvious outliers (intervals outside [3/4, 4/3] * median).

        Returns:
            Current speed, or 0.0 if not enough calibration data is available.
        """
        if len(self._intervals) < self._min_intervals:
            return 0.0

        # Discard obvious outliers using the median.
        median = statistics.median(self._intervals)
        below = 3 / 4 * median
        above = 4 / 3 * median
        clean_intervals = [i for i in self._intervals if below < i < above]

        if len(clean_intervals) < self._min_intervals:
            return 0.0

        avg_interval = statistics.mean(clean_intervals)
        if avg_interval <= 0:
            return 0.0

        return self.length / avg_interval

    def predict_travel_time(self, distance: float) -> float:
        """
        Predict the travel time for an object to cover a certain distance.

        Args:
            distance: Distance to travel in millimeters (mm).

        Returns:
            Travel time in seconds (s), or 0.0 if not enough calibration data.
        """
        current_speed = self.speed
        if current_speed <= 0:
            return 0.0

        return distance / current_speed
