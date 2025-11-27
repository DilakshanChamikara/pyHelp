from machine import Pin
import time

# Onboard LED on Pico/Pico W
led = Pin("LED", Pin.OUT)

while True:
    led.on()       # Turn LED on
    time.sleep(1)  # Wait 1 second
    led.off()      # Turn LED off
    time.sleep(1)  # Wait 1 second
