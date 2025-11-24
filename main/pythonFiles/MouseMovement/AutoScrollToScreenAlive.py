import pyautogui
import time

# Customize your scroll speed and duration
scroll_amount = -100  # negative for scroll down, positive for scroll up
delay_seconds = 2     # time between scrolls
scroll_times = 20     # how many times to scroll

print("Auto-scrolling will start in 5 seconds. Move your mouse to the top-left corner to abort.")
time.sleep(5)

for i in range(scroll_times):
    pyautogui.scroll(scroll_amount)
    print(f"Scrolled {i + 1} time(s)")
    time.sleep(delay_seconds)
