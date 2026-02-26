#!/usr/bin/env python3

"""Send updates to 16 servo channels, for testing PWM hardware and cables."""

import adafruit_pca9685
import board
import busio

import servo_channel
import servo_controller


def main():
    step = 5
    angles = list(range(0, 180 + step, step))

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = adafruit_pca9685.PCA9685(i2c, address=0x41)
    controller = servo_controller.ServoController(pca=pca)
    servos = servo_channel.parse_ranges("A1:A16", controller)

    offset = 0
    print("Starting servo loop. Press Ctrl+C to stop.")
    try:
        while True:
            offset += 1
            for servo in servos:
                # Stagger the angles for each servo to distribute current draw.
                angle = angles[(offset + servo.channel * 4) % len(angles)]

                # Print debug info for the first cell.
                if servo.col == "A" and servo.row == 1:
                    print(f"Servo {servo} angle={angle:3d}", end="\r")

                servo.send_angle(angle)
    except KeyboardInterrupt:
        print("\nStopping servo demo...")
    finally:
        print("De-initializing controllers...")
        pca.deinit()


if __name__ == "__main__":
    main()
