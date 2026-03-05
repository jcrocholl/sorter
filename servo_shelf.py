from typing import Any, Callable
from servo_controller import ServoController
from servo_channel import ServoChannel, parse_ranges


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
