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
    ) -> None:
        """Initialize the sorting shelf with a configuration.

        The config is a dictionary where each key is the I2C address (int) of a
        PCA9685 controller, and each value is a space-separated string of ranges
        for servo channels (e.g., "A1:A10 B1:B5"). Servo channels are assigned
        sequentially starting from 0 for each controller.
        """
        self.controllers: dict[int, ServoController] = {}
        self.servos: dict[str, ServoChannel] = {}

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
