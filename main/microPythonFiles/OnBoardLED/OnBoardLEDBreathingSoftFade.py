from machine import Pin
import time

led = Pin("LED", Pin.OUT)

while True:
    # Fade in
    for i in range(100):
        led.on()
        time.sleep(i / 5000)
        led.off()
        time.sleep((100 - i) / 5000)
    # Fade out
    for i in range(100):
        led.on()
        time.sleep((100 - i) / 5000)
        led.off()
        time.sleep(i / 5000)
