import bisect
import threading
import time
from typing import Any, Callable

from servo_channel import ServoChannel, parse_ranges
from servo_controller import ServoController


class ServoShelf:
    """Holds all ServoChannel objects for the entire sorting shelf."""

    def __init__(
        self,
        config: dict[int, str],
        pca_factory: Callable[[int], Any],
        conveyor_belt: Any,
        brick_mapping: Any,
    ) -> None:
        """Initialize the sorting shelf with a configuration.

        The config is a dictionary where each key is the I2C address (int) of a
        PCA9685 controller, and each value is a space-separated string of ranges
        for servo channels (e.g., "A1:A10 B1:B5"). Servo channels are assigned
        sequentially starting from 0 for each controller.
        """
        self.controllers: dict[int, ServoController] = {}
        self.servos: dict[str, ServoChannel] = {}
        self.conveyor_belt = conveyor_belt
        self.brick_mapping = brick_mapping

        for address, range_str in config.items():
            pca = pca_factory(address)
            controller = ServoController(pca)
            self.controllers[address] = controller

            # Parse the ranges and add them to the servos dict.
            channels = parse_ranges(range_str, controller)
            for servo in channels:
                label = str(servo)
                assert label not in self.servos, f"Duplicate servo label: {label}"
                self.servos[label] = servo

        self._queue: list[tuple[float, str, float]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_event(self, timestamp: float, label: str, angle: float) -> None:
        """Add a servo movement to the sorted planner queue."""
        with self._lock:
            bisect.insort(self._queue, (timestamp, label, angle))

    def start(self) -> None:
        """Start the background thread to process the queue."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _process_queue(self) -> None:
        """Background thread loop to process the queue."""
        while not self._stop_event.is_set():
            now = time.time()
            event = None

            with self._lock:
                if self._queue and self._queue[0][0] <= now:
                    event = self._queue.pop(0)

            if event:
                timestamp, label, angle = event
                if label in self.servos:
                    self.servos[label].send_angle(angle)
                else:
                    print(f"Warning: Servo label '{label}' not found in shelf.")
            else:
                # Sleep a bit to avoid busy waiting.
                time.sleep(0.01)

    def on_brick_recognized(
        self,
        timestamp: float,
        brick_class: str,
    ) -> None:
        """Handle a recognized brick by scheduling kicker and flap movements.

        Args:
            timestamp: Time when the brick was recognized.
            brick_class: The class name of the recognized brick.
        """
        cell_label = self.brick_mapping.get_cell(brick_class)
        # Column is the first character of the cell label ('A1' => 'A').
        column = cell_label[0]
        kicker_label = column + "0"

        distance = self.conveyor_belt.get_kicker_distance(kicker_label)
        travel_time = self.conveyor_belt.predict_travel_time(distance)
        if travel_time <= 0:
            return

        kick_time = timestamp + travel_time

        # 1. Open the shelf box flap slightly before the brick arrives.
        self.add_event(kick_time - 0.1, cell_label, 90.0)  # Open

        # 2. Kick the brick at the right moment.
        self.add_event(kick_time, kicker_label, 45.0)  # Kick
        # TODO: The kicker servo needs to be reset, but this can only happen
        # during a blank space on the conveyor belt, otherwise it would
        # kick the wrong parts in the wrong direction.

        # 3. Close the flap after the brick has fallen into the drawer.
        self.add_event(kick_time + 0.5, cell_label, 0.0)  # Close
