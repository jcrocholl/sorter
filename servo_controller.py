from typing import Any

DEBUG = False


class ServoController:
    """Interface for a multi-channel PWM servo controller."""

    def __init__(
        self,
        pca: Any,  # Real PCA9685 or MagicMock.
        num_channels: int = 16,
        frequency: float = 50.0,  # Hz
    ) -> None:
        self.pca = pca
        self.num_channels = num_channels
        self.frequency = frequency
        self.send_frequency()

    def send_frequency(self) -> None:
        """Transmit frequency setting over I2C."""
        assert 40 < self.frequency < 200  # Hz
        self.pca.frequency = self.frequency

    def send_pwm_regs(self, channel: int, on: int, off: int) -> None:
        """Transmit new 12-bit values for on/off duty cycle over I2C."""
        assert 0 <= channel < self.num_channels
        assert 0 <= on < 0xFFF
        assert 0 <= off < 0xFFF
        self.pca.pwm_regs[channel] = (on, off)

    def send_angle(self, channel: int, angle: float) -> None:
        """Send a new angle for this servo."""
        pulse_usec = 600 + 1800 * angle / 180
        period_usec = 1_000_000 / self.frequency
        # Distribute rising edges across the duty cycle.
        on = int(0xFFF * channel / 16) % 0xFFF
        # Tested on 2026-02-22: it's acceptable for the off time to wrap
        # around into the next cycle (as long as 0 <= off < 0xFFF).
        off = int(on + 0xFFF * pulse_usec / period_usec) % 0xFFF
        if DEBUG:
            print(
                f"pulse={pulse_usec}us period={period_usec}us"
                f" channel={channel} on={on} off={off} 0xFFF={0xFFF}"
            )
        self.send_pwm_regs(channel, on, off)
