from machine import Pin
import time

led = Pin("LED", Pin.OUT)

def dot():
    led.on()
    time.sleep(0.2)
    led.off()
    time.sleep(0.2)

def dash():
    led.on()
    time.sleep(0.6)
    led.off()
    time.sleep(0.2)

while True:
    # S (...)
    dot(); dot(); dot()
    time.sleep(0.6)
    # O (---)
    dash(); dash(); dash()
    time.sleep(0.6)
    # S (...)
    dot(); dot(); dot()
    time.sleep(1.5)
