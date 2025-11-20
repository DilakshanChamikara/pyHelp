import pyautogui
import time
import random

def keep_mouse_moving():
    try:
        while True:
            # Move the mouse randomly within ±50 pixels
            x_move = random.randint(-50, 50)
            y_move = random.randint(-50, 50)
            pyautogui.moveRel(x_move, y_move, duration=0.2)

            # Wait between 10 and 60 seconds2
            wait_time = random.randint(10, 60)
            print(f"Moved mouse. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
    except KeyboardInterrupt:
        print("Stopped by user.")

keep_mouse_moving()
