import pyautogui
import time
import random

def keep_screen_alive():
    try:
        while True:
            # Random small move to avoid suspicion
            x_move = random.randint(-100, 100)
            y_move = random.randint(-100, 100)
            pyautogui.moveRel(x_move, y_move, duration=0.2)

            # Random click (optional, can be removed)
            pyautogui.click()

            # Wait between 10–60 seconds
            wait_time = random.randint(10, 60)
            print(f"Clicked and moved. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
    except KeyboardInterrupt:
        print("Script stopped by user.")

# Run the function
keep_screen_alive()
